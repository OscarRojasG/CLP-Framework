import torch
import torch.nn as nn
from models.base.transformer import Transformer

# Variante de CLPTransformer con Pre-LN en todo el modelo:
#     x = x + sublayer(norm(x))
# En CLPTransformer el encoder y el contenedor usan Post-LN (x = norm(x + sublayer(x)))
# mientras que el decoder usa Pre-LN. Aquí todas las rutas siguen la misma convención,
# y cada ruta termina con una LayerNorm final antes de ser usada como memoria.

class InventoryAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_dim_multiplier, dropout):
        super().__init__()

        # ==========================================================
        # PASO 1 (Izquierda): Inventario -> Lee Bloques
        # ==========================================================
        self.norm_inv = nn.LayerNorm(d_model)       # Q (inventario)
        self.norm1_blocks = nn.LayerNorm(d_model)   # K/V del paso 1 y Q del paso 2 (bloques)
        self.cross_attn_inv = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # ==========================================================
        # PASO 2 (Derecha): Bloques -> Leen Inventario -> MLP
        # ==========================================================
        self.norm_inv_kv = nn.LayerNorm(d_model)    # K/V del paso 2 (inventario actualizado)
        self.cross_attn_blocks = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm2_blocks = nn.LayerNorm(d_model)
        self.mlp_blocks = nn.Sequential(
            nn.Linear(d_model, d_model * ff_dim_multiplier),
            nn.ReLU(),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x_blocks, x_inv, block_padding_mask=None):
        """
        x_blocks: (Batch, N_blocks, d_model)
        x_inv: (Batch, 1, d_model)
        block_padding_mask: Máscara booleana (Batch, N_blocks). True indica padding/inválido.
        """
        # Los bloques no cambian durante la fase 1, así que la misma normalización
        # sirve como K/V de la fase 1 y como Q de la fase 2
        normed_blocks = self.norm1_blocks(x_blocks)

        # ---------------------------------------------------------
        # FASE 1: El token de inventario absorbe el estado global
        # ---------------------------------------------------------
        # Q = Inventario, K = V = Bloques
        attn_inv_out, _ = self.cross_attn_inv(
            query=self.norm_inv(x_inv),
            key=normed_blocks,
            value=normed_blocks,
            key_padding_mask=block_padding_mask
        )
        x_inv = x_inv + attn_inv_out

        # ---------------------------------------------------------
        # FASE 2: Los bloques se actualizan usando el estado global
        # ---------------------------------------------------------
        # Q = Bloques, K = V = Inventario
        normed_inv = self.norm_inv_kv(x_inv)
        attn_blocks_out, _ = self.cross_attn_blocks(
            query=normed_blocks,
            key=normed_inv,
            value=normed_inv
        )
        x_blocks = x_blocks + attn_blocks_out

        # Procesamiento individual (Feed Forward)
        x_blocks = x_blocks + self.mlp_blocks(self.norm2_blocks(x_blocks))

        return x_blocks, x_inv

class CLPTransformerFix(Transformer):
    def __init__(self, block_dim, action_dim, placed_dim, d_model=64, nhead=4, encoder_layers=3, decoder_layers=3, ff_dim_multiplier=3, dropout=0.1):
        super().__init__(
            block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim,
            d_model=d_model, nhead=nhead, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
            ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        self.d_model = d_model
        self.decoder_layers = decoder_layers

        # ======================================================
        # --- 1. RUTA DEL ENCODER (CATÁLOGO / INVENTARIO) ---
        # ======================================================
        self.block_proj = nn.Linear(block_dim, d_model)

        # Token latente inicial (Inventory Token)
        self.inv_query_token = nn.Parameter(torch.randn(1, 1, d_model))

        # El bloque iterativo x N del diagrama
        self.inventory_layers = nn.ModuleList([
            InventoryAttentionBlock(d_model, nhead, ff_dim_multiplier, dropout)
            for _ in range(encoder_layers)
        ])

        # Normalización final de la salida del encoder (Pre-LN)
        self.final_enc_norm = nn.LayerNorm(d_model)

        # ======================================================
        # --- 2. RUTA DE ACCIONES CANDIDATO (IZQUIERDA) ---
        # ======================================================
        # Proyección para métricas operacionales: [loss, cs] -> Entrada fija de dimensión 2
        self.action_feat_proj = nn.Linear(action_dim, d_model)

        # MLP de Fusión para Query definitiva: Concatena block_emb (d_model) + action_feat_emb (d_model)
        self.query_fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model * ff_dim_multiplier),
            nn.ReLU(),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.Dropout(dropout)
        )

        # --- 3. RUTA DE ENTORNO: CONTENEDOR ENRIQUECIDO ---
        self.placed_proj = nn.Linear(placed_dim, d_model)

        # Token único que representa el contenedor global
        self.container_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.empty_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Atención para que el contenedor recolecte información de los bloques
        self.container_query_norm = nn.LayerNorm(d_model)
        self.container_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        # Normalización final del contenedor antes de usarlo como memoria del decoder
        self.container_norm = nn.LayerNorm(d_model)

        # --- 4. DECODER: Acciones atienden al Contenedor ---
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
            for _ in range(decoder_layers)
        ])
        self.decoder_ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * ff_dim_multiplier),
                nn.ReLU(),
                nn.Linear(d_model * ff_dim_multiplier, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(decoder_layers)
        ])
        self.attn_dropout = nn.Dropout(dropout)

        self.dec_norm1_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(decoder_layers)])
        self.dec_norm2_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(decoder_layers)])
        self.final_dec_norm = nn.LayerNorm(d_model)

        # ======================================================
        # --- 5. CABEZAL DE SALIDA (LOGITS DE ENERGÍA) ---
        # ======================================================
        self.energy_proj = nn.Linear(d_model, 1)

    def encode(self, block_features):
        B, N_blocks, _ = block_features.shape

        block_padding_mask = torch.all(block_features == -1.0, dim=-1)

        # 1. Proyección Lineal Inicial
        x_blocks = self.block_proj(block_features)

        # 2. Inicializar el token de inventario latente
        x_inv = self.inv_query_token.expand(B, -1, -1)

        # 3. Bucle iterativo (El Inventario lee -> Los Bloques leen)
        for layer in self.inventory_layers:
            x_blocks, x_inv = layer(x_blocks, x_inv, block_padding_mask)

        return self.final_enc_norm(x_blocks),

    def decode(self, memory, action_blocks, action_features, placed_features):
        B = memory.shape[0]

        # --- 1. RUTA IZQUIERDA: Preparar Acciones (Sin Self-Attention) ---
        action_mask = (action_blocks != -1)
        action_idx_clamped = action_blocks.clamp(min=0)
        block_emb_action = torch.gather(memory, 1, action_idx_clamped.unsqueeze(-1).expand(-1, -1, self.d_model))
        action_feat_emb = self.action_feat_proj(action_features)

        # queries base [B, Na, d_model]
        queries = self.query_fusion_mlp(torch.cat([block_emb_action, action_feat_emb], dim=-1))

        # --- 2. RUTA DERECHA: Contenedor Enriquecido ---
        placed_mask = ~torch.all(placed_features == -1.0, dim=-1) # [B, Np]
        placed_sequence = self.placed_proj(placed_features)

        # INYECCIÓN: Crear el token de "vacío" y concatenarlo al inicio
        empty_t = self.empty_token.expand(B, 1, -1)
        placed_sequence = torch.cat([empty_t, placed_sequence], dim=1)

        # Actualizar máscara: el nuevo token en la pos 0 debe ser False (no ignorar)
        full_mask = torch.cat([torch.zeros(B, 1, device=placed_mask.device, dtype=torch.bool), ~placed_mask], dim=1)

        # El contenedor (token único) atiende a todos los bloques (incluyendo el empty_token)
        container = self.container_token.expand(B, -1, -1)
        container_attn_out, _ = self.container_attn(
            query=self.container_query_norm(container),
            key=placed_sequence,
            value=placed_sequence,
            key_padding_mask=full_mask
        )
        container_enriched = self.container_norm(container + container_attn_out)

        # --- 3. CROSS-ATTENTION: Acciones atienden al Contenedor ---
        for i in range(self.decoder_layers):
            normed_queries = self.dec_norm1_layers[i](queries)
            # Acciones (Q) atienden al contenedor (K, V)
            attn_output, _ = self.cross_attn_layers[i](
                query=normed_queries,
                key=container_enriched,
                value=container_enriched
            )
            queries = queries + self.attn_dropout(attn_output)

            normed_queries_2 = self.dec_norm2_layers[i](queries)
            queries = queries + self.decoder_ff_layers[i](normed_queries_2)

        logits = self.energy_proj(self.final_dec_norm(queries)).squeeze(-1)
        return logits.masked_fill(~action_mask, -1e9)

import torch
import torch.nn as nn
from models.base.transformer import Transformer
import torch
import torch.nn as nn

class InventoryAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_dim_multiplier, dropout):
        super().__init__()
        
        # ==========================================================
        # PASO 1 (Izquierda): Inventario -> Lee Bloques
        # ==========================================================
        self.cross_attn_inv = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_inv = nn.LayerNorm(d_model)

        # ==========================================================
        # PASO 2 (Derecha): Bloques -> Leen Inventario -> MLP
        # ==========================================================
        self.cross_attn_blocks = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1_blocks = nn.LayerNorm(d_model)
        
        self.mlp_blocks = nn.Sequential(
            nn.Linear(d_model, d_model * ff_dim_multiplier),
            nn.ReLU(),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.Dropout(dropout)
        )
        self.norm2_blocks = nn.LayerNorm(d_model)

    def forward(self, x_blocks, x_inv, block_padding_mask=None):
        """
        x_blocks: (Batch, N_blocks, d_model)
        x_inv: (Batch, 1, d_model)
        block_padding_mask: Máscara booleana (Batch, N_blocks). True indica padding/inválido.
        """
        
        # ---------------------------------------------------------
        # FASE 1: El token de inventario absorbe el estado global
        # ---------------------------------------------------------
        # Q = Inventario, K = V = Bloques
        # CRÍTICO: Usamos 'key_padding_mask' para evitar que el resumen 
        # absorba ruido matemático de los elementos ignorados.
        attn_inv_out, _ = self.cross_attn_inv(
            query=x_inv, 
            key=x_blocks, 
            value=x_blocks,
            key_padding_mask=block_padding_mask
        )
        # El token ahora contiene el resumen exacto de ESTE estado
        x_inv = self.norm_inv(x_inv + attn_inv_out)

        # ---------------------------------------------------------
        # FASE 2: Los bloques se actualizan usando el estado global
        # ---------------------------------------------------------
        # Q = Bloques, K = V = Inventario
        # Como el inventario es un solo token válido (no tiene padding), 
        # no necesitamos pasar la máscara aquí.
        attn_blocks_out, _ = self.cross_attn_blocks(
            query=x_blocks, 
            key=x_inv, 
            value=x_inv
        )
        x_blocks = self.norm1_blocks(x_blocks + attn_blocks_out)
        
        # Procesamiento individual (Feed Forward)
        mlp_out = self.mlp_blocks(x_blocks)
        x_blocks = self.norm2_blocks(x_blocks + mlp_out)

        return x_blocks, x_inv

class CLPTransformer(Transformer):
    def __init__(self, block_dim, action_dim, placed_dim, d_model=64, nhead=4, num_layers=3, ff_dim_multiplier=3, dropout=0.1):
        super().__init__(
            block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        self.d_model = d_model
        self.num_layers = num_layers

        # ======================================================
        # --- 1. RUTA DEL ENCODER (CATÁLOGO / INVENTARIO) ---
        # ======================================================
        # Equivale a la caja inferior "Linear" en tu diagrama
        self.block_proj = nn.Linear(block_dim, d_model)
        
        # Token latente inicial (Inventory Token)
        self.inv_query_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # El bloque iterativo x N del diagrama
        self.inventory_layers = nn.ModuleList([
            InventoryAttentionBlock(d_model, nhead, ff_dim_multiplier, dropout)
            for _ in range(num_layers)
        ])

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
        self.container_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.container_norm = nn.LayerNorm(d_model)

        # --- 4. DECODER: Acciones atienden al Contenedor ---
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.decoder_ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * ff_dim_multiplier),
                nn.ReLU(),
                nn.Linear(d_model * ff_dim_multiplier, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        self.attn_dropout = nn.Dropout(dropout)
        
        self.dec_norm1_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dec_norm2_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.final_dec_norm = nn.LayerNorm(d_model)

        # ======================================================
        # --- 5. CABEZAL DE SALIDA (LOGITS DE ENERGÍA) ---
        # ======================================================
        self.energy_proj = nn.Linear(d_model, 1)

    def encode(self, block_features):
        B, N_blocks, _ = block_features.shape
        
        # Calculamos la máscara internamente, tal como tenías en tu código original
        block_padding_mask = torch.all(block_features == -1.0, dim=-1)
        
        # 1. Proyección Lineal Inicial
        x_blocks = self.block_proj(block_features)

        # 2. Inicializar el token de inventario latente
        x_inv = self.inv_query_token.expand(B, -1, -1)

        # 3. Bucle iterativo (El Inventario lee -> Los Bloques leen)
        for layer in self.inventory_layers:
            x_blocks, x_inv = layer(x_blocks, x_inv, block_padding_mask)
        
        return x_blocks,

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
        container_enriched, _ = self.container_attn(
            query=container, 
            key=placed_sequence, 
            value=placed_sequence, 
            key_padding_mask=full_mask
        )
        container_enriched = self.container_norm(container + container_enriched)

        # --- 3. CROSS-ATTENTION: Acciones atienden al Contenedor ---
        for i in range(self.num_layers):
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
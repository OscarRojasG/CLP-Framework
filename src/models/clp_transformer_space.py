import torch
import torch.nn as nn
from models.base.mlp_encoder import MLPEncoder
from models.base.transformer import Transformer

class CLPTransformer(Transformer):
    # NUEVO: Se añade space_dim a la firma para inicializar space_proj
    def __init__(self, block_dim, action_dim, placed_dim, space_dim, d_model=64, nhead=4, num_layers=3, ff_dim_multiplier=3, dropout=0.1):
        # Mantenemos el super por compatibilidad, aunque internamente ignoramos las dimensiones sobrantes
        super().__init__(
            block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim, space_dim=space_dim,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        self.d_model = d_model
        self.num_layers = num_layers

        # ======================================================
        # --- 1. RUTA DEL ENCODER (CATÁLOGO / INVENTARIO) ---
        # ======================================================
        self.block_proj = nn.Linear(block_dim, d_model)
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)

        # Pooling de Inventario
        self.inv_query_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.inv_pooling_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.summary_proj = nn.Linear(d_model, d_model)
        self.summary_dropout = nn.Dropout(dropout)
        self.norm_enrich = nn.LayerNorm(d_model)

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

        # NUEVO: Reemplazamos el container_token estático por una proyección de space_features
        self.space_proj = nn.Linear(space_dim, d_model)

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
        block_padding_mask = torch.all(block_features == -1.0, dim=-1)

        x = self.block_proj(block_features)
        raw_memory = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)

        query = self.inv_query_token.expand(B, -1, -1)
        inv_summary, _ = self.inv_pooling_attn(
            query=query, key=raw_memory, value=raw_memory, key_padding_mask=block_padding_mask
        )
        contextual_modifier = self.summary_dropout(self.summary_proj(inv_summary))
        enriched_memory = self.norm_enrich(raw_memory + contextual_modifier)
        return enriched_memory,

    def decode(self, memory, action_blocks, action_features, placed_features, space_features):
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

        # Aseguramos que tenga la dimensión de secuencia requerida [B, 1, d_model]
        space_emb = self.space_proj(space_features)
        space_emb = space_emb.unsqueeze(1)
        
        # El espacio atiende a todos los bloques (incluyendo el empty_token)
        space_enriched, _ = self.container_attn(
            query=space_emb,
            key=placed_sequence,
            value=placed_sequence,
            key_padding_mask=full_mask
        )
        space_enriched = self.container_norm(space_emb + space_enriched)

        # --- 3. CROSS-ATTENTION: Acciones atienden al Espacio ---
        for i in range(self.num_layers):
            normed_queries = self.dec_norm1_layers[i](queries)
            # Acciones (Q) atienden al espacio (K, V)
            attn_output, _ = self.cross_attn_layers[i](
                query=normed_queries,
                key=space_enriched,
                value=space_enriched
            )
            queries = queries + self.attn_dropout(attn_output)

            normed_queries_2 = self.dec_norm2_layers[i](queries)
            queries = queries + self.decoder_ff_layers[i](normed_queries_2)

        logits = self.energy_proj(self.final_dec_norm(queries)).squeeze(-1)
        return logits.masked_fill(~action_mask, -1e9)
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.mlp_encoder import MLPEncoder
from models.base.transformer import Transformer

class CLPTransformer(Transformer):
    def __init__(self, block_dim, action_dim, space_dim, placed_dim, d_model=64, nhead=4, num_layers=3, ff_dim_multiplier=3, dropout=0.1):
        super().__init__(
            block_dim=block_dim, action_dim=action_dim, space_dim=space_dim, placed_dim=placed_dim,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        self.d_model = d_model
        self.num_layers = num_layers

        # --- ENCODER (igual que antes) ---
        self.block_proj = nn.Linear(block_dim, d_model)
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        self.inv_query_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.inv_pooling_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.summary_proj = nn.Linear(d_model, d_model)
        self.summary_dropout = nn.Dropout(dropout)
        self.norm_enrich = nn.LayerNorm(d_model)

        # --- PROYECCIONES MLP para coordenadas ---
        def coord_mlp(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, d_model * ff_dim_multiplier),
                nn.ReLU(),
                nn.Linear(d_model * ff_dim_multiplier, d_model),
                nn.Dropout(dropout)
            )

        self.space_mlp   = coord_mlp(6)   # espacio: 6 coords
        self.placed_mlp  = coord_mlp(6)   # placed blocks: 6 coords
        self.action_mlp  = coord_mlp(6)   # acciones: 6 coords

        # --- CROSS-ATTENTION 1: espacio -> placed blocks ---
        self.space_cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.space_norm1 = nn.LayerNorm(d_model)
        self.space_norm2 = nn.LayerNorm(d_model)
        self.space_ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_dim_multiplier),
            nn.ReLU(),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.Dropout(dropout)
        )

        # Token vacío de seguridad para placed blocks
        self.empty_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- CROSS-ATTENTION 2: acciones -> espacio contextualizado ---
        self.action_cross_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.action_ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * ff_dim_multiplier),
                nn.ReLU(),
                nn.Linear(d_model * ff_dim_multiplier, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        self.action_norm1_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.action_norm2_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        # --- CABEZAL FINAL ---
        # Concatena: acción enriquecida (d_model) + block_emb (d_model) + action_features (action_dim)
        self.action_feat_proj = nn.Linear(action_dim, d_model)

        self.final_mlp = nn.Sequential(
            nn.Linear(2*d_model, d_model * ff_dim_multiplier),
            nn.ReLU(),
            nn.Linear(d_model * ff_dim_multiplier, 1),
            nn.Dropout(dropout)
        )

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

    def decode(self, memory, action_blocks, action_coords, placed_coords, space_coords, action_features):
        B = memory.shape[0]
        Na = action_blocks.shape[1]

        action_mask = (action_blocks != -1)
        action_idx_clamped = action_blocks.clamp(min=0)

        # --- 1. EMBEDDINGS DE COORDENADAS ---
        # Espacio: (B, 1, d_model)
        space_emb = self.space_mlp(space_coords).unsqueeze(1)

        # Placed blocks: (B, Np, d_model) con token vacío de seguridad
        placed_mask = torch.all(placed_coords == -1, dim=-1)  # (B, Np)
        placed_coords_clean = placed_coords.clone()
        placed_coords_clean[placed_mask] = 0.0
        placed_emb = self.placed_mlp(placed_coords_clean)     # (B, Np, d_model)
        placed_emb = placed_emb * (~placed_mask).unsqueeze(-1).float()

        dummy_token = self.empty_token.expand(B, -1, -1)
        placed_with_dummy = torch.cat([dummy_token, placed_emb], dim=1)  # (B, 1+Np, d_model)
        dummy_mask = torch.zeros((B, 1), dtype=torch.bool, device=placed_coords.device)
        placed_key_mask = torch.cat([dummy_mask, placed_mask], dim=1)    # (B, 1+Np)

        # Acciones: (B, Na, d_model)
        action_coords_clean = action_coords.clone()
        action_coords_clean[~action_mask] = 0.0
        action_emb = self.action_mlp(action_coords_clean)     # (B, Na, d_model)

        # Block embeddings desde memory
        block_emb = torch.gather(memory, 1, action_idx_clamped.unsqueeze(-1).expand(-1, -1, self.d_model))

        # --- 2. CROSS-ATTENTION 1: espacio -> placed blocks ---
        normed_space = self.space_norm1(space_emb)
        space_ctx, _ = self.space_cross_attn(
            query=normed_space,
            key=placed_with_dummy,
            value=placed_with_dummy,
            key_padding_mask=placed_key_mask
        )

        space_ctx = space_emb + space_ctx
        space_ctx = space_ctx + self.space_ff(self.space_norm2(space_ctx))  # (B, 1, d_model)

        # --- 3. CROSS-ATTENTION 2: acciones -> espacio contextualizado ---
        queries = action_emb + block_emb
        for i in range(self.num_layers):
            normed_q = self.action_norm1_layers[i](queries)
            attn_out, _ = self.action_cross_layers[i](
                query=normed_q,
                key=space_ctx,
                value=space_ctx
            )
            queries = queries + attn_out
            queries = queries + self.action_ff_layers[i](self.action_norm2_layers[i](queries))

        queries = self.final_norm(queries)  # (B, Na, d_model)

        # --- 4. CABEZAL FINAL ---
        # Concatenar: acción enriquecida + block embedding + action_features
        feat_emb = self.action_feat_proj(action_features)           # (B, Na, d_model)
        combined = torch.cat([queries, feat_emb], dim=-1)           # (B, Na, 2*d_model)
        logits = self.final_mlp(combined).squeeze(-1)                         # (B, Na)

        logits = logits.masked_fill(~action_mask, float('-inf'))
        return logits
import torch
import torch.nn as nn
from models.base.attention import CrossAttentionDecoder
from models.base.mlp_encoder import MLPEncoder
from models.base.transformer import Transformer


class CostPredictorTransformer(Transformer):
    def __init__(self, block_dim, placed_dim, space_dim, d_model=256, nhead=8, num_layers=3, ff_dim_multiplier=4, dropout=0.1):
        super().__init__(
            block_dim=block_dim,
            placed_dim=placed_dim,
            space_dim=space_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier,
            dropout=dropout
        )
        self.d_model = d_model

        # Componentes principales
        self.block_proj = nn.Linear(block_dim, d_model)
        self.placed_proj = nn.Linear(placed_dim, d_model)
        self.space_proj = nn.Linear(space_dim, d_model)
        
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        self.placed_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        
        # Proyección intermedia para bloques colocados (bloque + features extra)
        self.final_placed_proj = nn.Linear(d_model * 2, d_model)
        
        # Atención cruzada principal: Espacio -> Bloques Colocados
        self.cross_att = CrossAttentionDecoder(d_model, nhead, ff_dim_multiplier, dropout, num_layers)
        
        # Nueva Atención Cruzada: Contexto -> Memoria Global de Bloques
        # Usamos 1 sola capa para esta fusión final para mantenerlo eficiente y ligero
        self.fussion_cross_att = CrossAttentionDecoder(d_model, nhead, ff_dim_multiplier, dropout, num_layers=1)
        
        # Cabeza de regresión minimalista (Proyección directa a un único escalar de estado)
        self.regressor = nn.Linear(d_model, 1)
        
        self.empty_token = nn.Parameter(torch.randn(1, 1, d_model))

    def encode(self, block_features):
        B, N_blocks, _ = block_features.shape

        # Proyectar y Codificar
        x = self.block_proj(block_features) # [B, N, d_model]
        x = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)

        return (x, )

    def decode(self, memory, placed_blocks, placed_features, space_features):
        B = memory.shape[0]
        
        # 1. EMBEDDINGS BLOQUES COLOCADOS
        placed_mask = placed_blocks != -1 
        placed_idx = placed_blocks.clamp(min=0)
        block_emb = torch.gather(memory, 1, placed_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        placed_features = self.placed_proj(placed_features)
        placed_extra = self.placed_encoder(placed_features)
        placed_emb = self.final_placed_proj(torch.cat([block_emb, placed_extra], dim=-1))
        
        empty_tokens = self.empty_token.expand(B, -1, -1)
        placed_emb_augmented = torch.cat([empty_tokens, placed_emb], dim=1)
        token_mask = torch.ones((B, 1), dtype=torch.bool, device=placed_mask.device)
        augmented_mask = torch.cat([token_mask, placed_mask], dim=1)

        # 3. EMBEDDING ESPACIO
        space_emb = self.space_proj(space_features).unsqueeze(1)  # [B, 1, d_model]

        # 4. ATENCIÓN ESPACIO → BLOQUES COLOCADOS (Generación del contexto enriquecido)
        ctx = self.cross_att(space_emb, placed_emb_augmented, placed_emb_augmented, augmented_mask) # [B, 1, d_model]
        
        # 5. ATENCIÓN CRUZADA DIRECTA (Contexto del espacio interroga a la memoria global)
        # Q = ctx [B, 1, d_model]
        # K, V = memory [B, N_blocks, d_model]
        # Al no pasar máscara aquí, permitimos que el contexto evalúe toda la memoria libre disponible
        fused_state = self.fussion_cross_att(ctx, memory, memory) # [B, 1, d_model]
        
        # 6. REDUCCIÓN Y REGRESIÓN FINAL
        fused_state = fused_state.squeeze(1) # [B, d_model]
        state_prediction = self.regressor(fused_state) # [B, 1]

        return state_prediction.view(-1)
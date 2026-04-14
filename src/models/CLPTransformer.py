import torch
import torch.nn as nn
from models.base.attention import CrossAttentionDecoder
from models.base.mlp_encoder import MLPEncoder
from models.base.transformer import Transformer


class CLPTransformer(Transformer):
    def __init__(self, block_dim, action_dim, placed_dim, space_dim, d_model=256, nhead=8, num_layers=3, ff_dim_multiplier=4, dropout=0.1):
        super().__init__(
            block_dim=block_dim,
            action_dim=action_dim,
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
        self.action_proj = nn.Linear(action_dim, d_model)
        self.placed_proj = nn.Linear(placed_dim, d_model)
        self.space_proj = nn.Linear(space_dim, d_model)
        
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        self.action_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        self.placed_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        
        self.final_placed_proj = nn.Linear(2*d_model, d_model)
        self.final_action_proj = nn.Linear(2*d_model, d_model)
        
        self.cross_att = CrossAttentionDecoder(d_model, nhead, ff_dim_multiplier, dropout, num_layers)

        # Proyecciones para el Scaled Dot-Product
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        
        self.empty_token = nn.Parameter(torch.randn(1, 1, d_model))

    def encode(self, block_features):
        B, N_blocks, _ = block_features.shape

        # Proyectar y Codificar
        x = self.block_proj(block_features) # [B, N, d_model]
        x = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)

        return x

    def decode(self, memory, action_blocks, action_features, placed_blocks, placed_features, space_features):
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

        # 2. EMBEDDINGS ACCIONES
        action_mask = action_blocks != -1 
        action_idx = action_blocks.clamp(min=0)
        block_emb_action = torch.gather(memory, 1, action_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        action_features = self.action_proj(action_features)
        action_extra = self.action_encoder(action_features)
        action_cat = torch.cat([block_emb_action, action_extra], dim=-1)
        action_emb = self.final_action_proj(action_cat)  # [B, Na, d_model]

        # 3. EMBEDDING ESPACIO
        space_emb = self.space_proj(space_features).unsqueeze(1)  # [B, 1, d_model]

        # 4. ATENCIÓN ESPACIO → BLOQUES COLOCADOS (Generación del contexto enriquecido)
        ctx = self.cross_att(space_emb, placed_emb_augmented, placed_emb_augmented, augmented_mask)   

        # ---------------------------------------------------
        # 5. SCALED DOT-PRODUCT DIRECTO
        # ---------------------------------------------------
        # Queremos comparar el vector de contexto (Query) con cada acción (Key)
        # ctx: [B, 1, d_model]
        # action_emb: [B, Na, d_model]
        
        # Proyectamos al espacio de comparación
        q = self.q_proj(ctx)          # [B, 1, d_model]
        k = self.k_proj(action_emb)   # [B, Na, d_model]
        
        # Calculamos scores: [B, 1, d_model] @ [B, d_model, Na] -> [B, 1, Na]
        # Usamos .transpose(-1, -2) para que las dimensiones coincidan para matmul
        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model**0.5)
        
        # Quitamos la dimensión extra del query
        logits = logits.squeeze(1)    # [B, Na]

        # Las acciones de relleno deben ser -inf para el Softmax
        logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits
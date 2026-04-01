import torch
import torch.nn as nn
import math # Importante para el escalado
from models.base.attention import CrossAttentionBlock
from models.base.transformer import Transformer

class AggregationLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # Preparación de la máscara
        m = mask.unsqueeze(-1).float() 
        n_valid = m.sum(dim=1, keepdim=True).clamp(min=1)

        # Cálculo de Estadísticas (Solo válidos)
        mu = (x * m).sum(dim=1, keepdim=True) / n_valid # [B, 1, d_model]
        
        # Desviación Estándar
        diff = (x - mu) * m
        var = (diff**2).sum(dim=1, keepdim=True) / n_valid
        std = torch.sqrt(var + 1e-6) # [B, 1, d_model]

        # Enriquecimiento (Concatenación)
        z = torch.cat([x, mu.expand_as(x), diff, std.expand_as(x)], dim=-1)

        # Proyección y Skip Connection
        out = self.fusion(z)
        return self.norm(out + x)

class MLPEncoder(nn.Module):
    def __init__(self, d_model, ff_dim_multiplier, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_dim_multiplier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Conexión residual + Normalización
        return self.norm(x + self.ffn(x))

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
        
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout)
        self.action_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout)
        self.placed_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout)
        
        self.block_agg = AggregationLayer(d_model, dropout)
        
        self.final_placed_proj = nn.Linear(2*d_model, d_model)
        self.final_action_proj = nn.Linear(2*d_model, d_model)
        
        self.ctx_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, ff_dim_multiplier, dropout)
            for _ in range(num_layers)
        ])

        # Proyecciones para el Scaled Dot-Product
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        
        self.empty_token = nn.Parameter(torch.randn(1, 1, d_model))

    def encode(self, block_features):
        B, N_blocks, _ = block_features.shape

        # Identificar bloques reales para la agregación
        mask = (block_features != -1).any(dim=-1) 

        # Proyectar y Codificar
        x = self.block_proj(block_features) # [B, N, d_model]
        x = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)
        
        # Enriquecer con contexto global
        E_src = self.block_agg(x, mask)

        return E_src

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
        ctx = space_emb
        for layer in self.ctx_layers:
            ctx = layer(
                query=ctx, 
                key=placed_emb_augmented, 
                value=placed_emb_augmented, 
                key_padding_mask=~augmented_mask 
            )
        # ctx queda como [B, 1, d_model]

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
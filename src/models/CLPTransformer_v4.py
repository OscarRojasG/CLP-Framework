import torch
import torch.nn as nn
from models.base.legacy import CrossAttentionBlock, SelfAttentionBlock
from models.base.transformer import Transformer

class MLPEncoder(nn.Module):
    def __init__(self, dim, d_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, X_src):
        return self.encoder(X_src)

class CLPTransformer(Transformer):
    def __init__(self, block_dim, action_dim, placed_dim, space_dim, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super().__init__(
            block_dim=block_dim,
            action_dim=action_dim,
            placed_dim=placed_dim,
            space_dim=space_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        self.d_model = d_model

        # Componentes principales
        self.block_encoder = MLPEncoder(block_dim, d_model)
        self.action_encoder = MLPEncoder(action_dim, d_model)
        self.placed_encoder = MLPEncoder(placed_dim, d_model)
        
        self.space_proj = nn.Linear(space_dim, d_model)
        self.final_placed_proj = nn.Linear(2*d_model, d_model)
        self.final_action_proj = nn.Linear(2*d_model, d_model)
        
        self.ctx_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        self.action_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        self.action_self_layers = nn.ModuleList([
            SelfAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        # Proyección final
        self.output = nn.Linear(d_model, 1)


    def encode(self, block_features):
        B, N_blocks, _ = block_features.shape

        E_src = self.block_encoder(
            block_features.view(-1, block_features.shape[-1])
        ).view(B, N_blocks, self.d_model)

        return E_src
    

    def decode(self, memory, action_blocks, action_features, placed_blocks, placed_features, space_features):
        """
        print("Action blocks shape", action_blocks.shape)
        print("Action features shape", action_features.shape)
        print("Placed blocks shape", placed_blocks.shape)
        print("Placed features shape", placed_features.shape)
        print("Space features shape", space_features.shape)
        """
        
        # ---------------------------------------------------
        # 1. EMBEDDINGS BLOQUES COLOCADOS
        # ---------------------------------------------------

        placed_mask = placed_blocks != -1  # [B, Np]

        placed_idx = placed_blocks.clamp(min=0)

        block_emb = torch.gather(
            memory,
            1,
            placed_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        placed_extra = self.placed_encoder(placed_features)

        placed_cat = torch.cat([block_emb, placed_extra], dim=-1)

        placed_emb = self.final_placed_proj(placed_cat)  # [B, Np, d]

        # ---------------------------------------------------
        # 2. EMBEDDINGS ACCIONES (Nuevo: Manejo de padding)
        # ---------------------------------------------------
        # Creamos máscara para identificar qué acciones son reales
        action_mask = action_blocks != -1  # [B, Na]
        
        # Clamp a 0 para el gather (los valores con mask False se ignorarán luego)
        action_idx = action_blocks.clamp(min=0)

        block_emb_action = torch.gather(
            memory,
            1,
            action_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        action_extra = self.action_encoder(action_features)
        action_cat = torch.cat([block_emb_action, action_extra], dim=-1)
        action_emb = self.final_action_proj(action_cat)  # [B, Na, d]

        # ---------------------------------------------------
        # 3. EMBEDDING ESPACIO
        # ---------------------------------------------------

        space_emb = self.space_proj(space_features).unsqueeze(1)  # [B,1,d]

        # ---------------------------------------------------
        # 4. ATENCIÓN ESPACIO → BLOQUES COLOCADOS
        # ---------------------------------------------------

        # 1. Identificar si hay bloques colocados
        has_placed = placed_mask.any(dim=-1, keepdim=True) # [B, 1]

        # 2. CREAR UNA MÁSCARA SEGURA: 
        # Si una fila es toda False (sin bloques), forzamos el primer elemento a ser True 
        # SOLO para que la capa de atención no explote.
        safe_placed_mask = placed_mask.clone()
        safe_placed_mask[~has_placed.squeeze(-1), 0] = True

        # 3. Aplicar las capas de atención
        ctx = space_emb

        for layer in self.ctx_layers:
            # 1. Llamamos al bloque completo (aprovechamos su forward)
            # Usamos safe_placed_mask para que nn.MultiheadAttention no explote
            new_ctx = layer(
                query=ctx, 
                key=placed_emb, 
                value=placed_emb, 
                key_padding_mask=~safe_placed_mask
            )

            # 2. Aplicamos el zero-out solo donde realmente no había bloques
            ctx = torch.where(
                has_placed.unsqueeze(-1),
                new_ctx,
                ctx # Mantenemos el space_emb original si no hay bloques colocados
            )

        # ---------------------------------------------------
        # 5. ATENCIÓN ACCIONES → CONTEXTO
        # ---------------------------------------------------

        attn_out = action_emb

        for layer in self.action_layers:
            attn_out = layer(
                query=attn_out,
                key=ctx,
                value=ctx
            )

        # ---------------------------------------------------
        # 6. PROYECCIÓN FINAL
        # ---------------------------------------------------

        logits = self.output(attn_out).squeeze(-1)  # [B,Na]
        
        # IMPORTANTE: Forzamos los logits de las acciones de padding a un valor muy bajo
        # para que no interfieran en el Softmax o la selección.
        logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits
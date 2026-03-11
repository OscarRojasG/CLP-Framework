import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.transformer import Transformer

### Igual a V2 pero con Self-Attention

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
        
        self.ctx_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.action_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.action_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Normalización y FF para contexto
        self.norm_ctx1 = nn.LayerNorm(d_model)
        self.norm_ctx2 = nn.LayerNorm(d_model)

        self.ff_ctx = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        # Normalización y FF para acciones→contexto
        self.norm_action1 = nn.LayerNorm(d_model)
        self.norm_action2 = nn.LayerNorm(d_model)

        self.ff_action = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        # Normalización y FF para self-attention de acciones
        self.norm_self1 = nn.LayerNorm(d_model)
        self.norm_self2 = nn.LayerNorm(d_model)

        self.ff_self = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

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
        # 2. EMBEDDINGS ACCIONES
        # ---------------------------------------------------

        action_idx = action_blocks

        block_emb = torch.gather(
            memory,
            1,
            action_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        action_extra = self.action_encoder(action_features)

        action_cat = torch.cat([block_emb, action_extra], dim=-1)

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

        # 3. Ejecutar la atención con la máscara segura
        ctx_attn_out, _ = self.ctx_cross_attn(
            query=space_emb,
            key=placed_emb,
            value=placed_emb,
            key_padding_mask=~safe_placed_mask  # Usamos la versión segura
        )

        # 4. LIMPIEZA: 
        # Para los batches que no tenían bloques, el resultado de ctx_attn_out 
        # es basura (atención al elemento 0 forzado). Lo ponemos a cero.
        ctx_attn_out = torch.where(has_placed.unsqueeze(-1), ctx_attn_out, torch.zeros_like(ctx_attn_out))

        # 5. El resto sigue igual
        ctx = self.norm_ctx1(space_emb + ctx_attn_out)
        ff_out = self.ff_ctx(ctx)
        ctx = self.norm_ctx2(ctx + ff_out)

        # ---------------------------------------------------
        # 5. ATENCIÓN ACCIONES → CONTEXTO
        # ---------------------------------------------------

        action_attn_out, _ = self.action_cross_attn(
            query=action_emb,
            key=ctx,
            value=ctx
        )

        attn_out = self.norm_action1(action_emb + action_attn_out)

        ff_out = self.ff_action(attn_out)

        attn_out = self.norm_action2(attn_out + ff_out)  # [B,Na,d]
        
        # ---------------------------------------------------
        # 5.1 SELF-ATTENTION ENTRE ACCIONES
        # ---------------------------------------------------

        self_attn_out, _ = self.action_self_attn(
            query=attn_out,
            key=attn_out,
            value=attn_out
        )

        attn_out = self.norm_self1(attn_out + self_attn_out)

        ff_out = self.ff_self(attn_out)

        attn_out = self.norm_self2(attn_out + ff_out)

        # ---------------------------------------------------
        # 6. PROYECCIÓN FINAL
        # ---------------------------------------------------

        logits = self.output(attn_out).squeeze(-1)  # [B,Na]

        probs = F.softmax(logits, dim=-1)

        return probs
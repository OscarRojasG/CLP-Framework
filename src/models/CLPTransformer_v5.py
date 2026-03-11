import torch
import torch.nn as nn
from models.base.attention import CrossAttentionBlock, SelfAttentionBlock
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
            block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim,
            space_dim=space_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout
        )
        self.d_model = d_model

        # Encoders y Proyecciones
        self.block_encoder = MLPEncoder(block_dim, d_model)
        self.action_encoder = MLPEncoder(action_dim, d_model)
        self.placed_encoder = MLPEncoder(placed_dim, d_model)
        self.space_proj = nn.Linear(space_dim, d_model)
        self.final_placed_proj = nn.Linear(2*d_model, d_model)
        self.final_action_proj = nn.Linear(2*d_model, d_model)
        
        # Capas de Atención
        # 1. Bloques -> Espacio
        self.ctx_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        
        # 2. Bloques <-> Bloques (NUEVA)
        self.placed_self_layers = nn.ModuleList([
            SelfAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])

        # 3. Acciones -> Contexto enriquecido
        self.action_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])

        self.output = nn.Linear(d_model, 1)

    def encode(self, block_features):
        B, N_blocks, _ = block_features.shape

        E_src = self.block_encoder(
            block_features.view(-1, block_features.shape[-1])
        ).view(B, N_blocks, self.d_model)

        return E_src

    def decode(self, memory, action_blocks, action_features, placed_blocks, placed_features, space_features):
        # ---------------------------------------------------
        # 1. EMBEDDINGS INICIALES
        # ---------------------------------------------------
        placed_mask = (placed_blocks != -1)  # [B, Np]
        placed_idx = placed_blocks.clamp(min=0)
        
        # Bloques colocados
        block_emb_placed = torch.gather(
            memory, 1, 
            placed_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        placed_extra = self.placed_encoder(placed_features)
        placed_emb = self.final_placed_proj(torch.cat([block_emb_placed, placed_extra], dim=-1))

        # Acciones
        action_extra = self.action_encoder(action_features)
        action_block_emb = torch.gather(
            memory, 1, 
            action_blocks.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        action_emb = self.final_action_proj(torch.cat([action_block_emb, action_extra], dim=-1))

        # Espacio global
        space_mem = self.space_proj(space_features).unsqueeze(1) # [B, 1, d]

        # ---------------------------------------------------
        # 2. ENRIQUECIMIENTO: BLOQUES COLOCADOS -> ESPACIO
        # ---------------------------------------------------
        enriched_placed = placed_emb
        has_placed = placed_mask.any(dim=1) # [B]

        if has_placed.any():
            # Solo procesamos la atención donde hay bloques
            for layer in self.ctx_layers:
                # Los bloques (Q) miran al espacio (K, V)
                new_placed = layer(query=enriched_placed, key=space_mem, value=space_mem)
                enriched_placed = torch.where(has_placed.view(-1, 1, 1), new_placed, enriched_placed)

            # Opcional: Auto-atención entre bloques (si quieres que interactúen entre sí)
            for layer in self.placed_self_layers:
                new_placed = layer(x=enriched_placed, mask=placed_mask)
                enriched_placed = torch.where(has_placed.view(-1, 1, 1), new_placed, enriched_placed)

        # ---------------------------------------------------
        # 3. ATENCIÓN ACCIONES -> ESPACIO
        # ---------------------------------------------------
        # Las acciones siempre interactúan con el espacio
        attn_out = action_emb
        # Nota: Usamos una capa distinta o la misma según tu arquitectura (aquí asumimos secuencial)
        for layer in self.action_layers[:1]: # Ejemplo: usar la primera capa para el espacio
            attn_out = layer(query=attn_out, key=space_mem, value=space_mem)

        # ---------------------------------------------------
        # 4. ATENCIÓN ACCIONES -> BLOQUES ENRIQUECIDOS
        # ---------------------------------------------------
        if has_placed.any():
            # Máscara de seguridad para el batch
            safe_mask = placed_mask.clone()
            safe_mask[~has_placed, 0] = True 
            
            for layer in self.action_layers[1:]: # El resto de las capas para los bloques
                new_attn = layer(
                    query=attn_out,
                    key=enriched_placed,
                    value=enriched_placed,
                    key_padding_mask=~safe_mask
                )
                attn_out = torch.where(has_placed.view(-1, 1, 1), new_attn, attn_out)

        # ---------------------------------------------------
        # 5. PROYECCIÓN FINAL
        # ---------------------------------------------------
        logits = self.output(attn_out).squeeze(-1)
        return logits
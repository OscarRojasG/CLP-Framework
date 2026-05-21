import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.transformer import Transformer


class ResidualBlock(nn.Module):
    """
    Bloque MLP residual que estabiliza el gradiente y ayuda 
    a mapear funciones de umbral rígido.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        x = F.relu(self.norm1(self.linear1(x)))
        x = self.dropout(x)
        x = self.norm2(self.linear2(x))
        return F.relu(x + res)


class CLPTransformer(Transformer):
    def __init__(self, block_dim=5, action_dim=2, space_dim=6, d_model=128, num_layers=3, dropout=0.1):
        super().__init__(
            block_dim=block_dim, action_dim=action_dim, space_dim=space_dim,
            d_model=d_model, nhead=1, num_layers=num_layers, ff_dim_multiplier=4, dropout=dropout
        )
        self.d_model = d_model
        
        # 🚨 NUEVA DIMENSIÓN DE ENTRADA CORREGIDA 🚨
        # Características de la Acción Candidata enriquecida: 
        # 5 (block_features) + 6 (space_features) + action_dim (features de la acción, ej: 2) = 13
        # Entrada al par relacional: 13 + 6 (placed_features) = 19
        input_dim = (block_dim + space_dim + action_dim) + space_dim 
        
        # Capa de proyección inicial adaptada a la nueva dimensión de entrada
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # Bloques residuales profundos compartidos
        self.relation_blocks = nn.ModuleList([
            ResidualBlock(d_model, dropout) for _ in range(num_layers)
        ])
        
        # Capa de salida con Pooling Dual
        self.output_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def encode(self, block_features):
        return block_features,

    def decode(self, memory, action_blocks, action_features, placed_features, space_features):
        B = memory.shape[0]
        Na = action_blocks.shape[1]
        N_placed = placed_features.shape[1]

        if len(space_features.shape) == 3:
            space_features = space_features.squeeze(1)
        elif len(space_features.shape) == 4:
            space_features = space_features.squeeze(1).squeeze(1)

        # --- STEP 1: CONSTRUIR GEOMETRÍA DE ACCIONES CANDIDATAS ENRIQUECIDAS ---
        action_mask = action_blocks != -1
        action_idx = action_blocks.clamp(min=0)
        
        # Reconstruimos las propiedades físicas del bloque de la acción [B, Na, 5]
        raw_block_action = torch.gather(memory, 1, action_idx.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))
        # Expandimos el espacio común [B, Na, 6]
        space_expanded = space_features.unsqueeze(1).expand(-1, Na, -1)
        
        # 🚨 TU MODIFICACIÓN AQUÍ: Concatenamos block_features + space_features + action_features 🚨
        # action_features tiene dimensiones [B, Na, action_dim]
        candidate_features = torch.cat([raw_block_action, space_expanded, action_features], dim=-1) # [B, Na, 13]

        # --- STEP 2: BROADCASTING CARTESIANO RELACIONAL ---
        # Expandimos los candidatos enriquecidos a: [B, Na, N_placed, 13]
        candidate_exp = candidate_features.unsqueeze(2).expand(-1, -1, N_placed, -1)
        # Expandimos bloques colocados a: [B, Na, N_placed, 6]
        placed_exp = placed_features.unsqueeze(1).expand(-1, Na, -1, -1)
        
        # Formamos la matriz relacional final [B, Na, N_placed, 19]
        pair_geometry = torch.cat([candidate_exp, placed_exp], dim=-1)

        # --- STEP 3: PROPAGACIÓN RESIDUAL PROFUNDA ---
        x = self.input_layer(pair_geometry) # [B, Na, N_placed, d_model]
        
        for block in self.relation_blocks:
            x = block(x)

        # --- STEP 4: ENMASCARAMIENTO DE BLOQUES VACÍOS (ANTI-RUIDO) ---
        placed_padding_mask = torch.any(placed_features < 0.0, dim=-1)
        mask_exp = placed_padding_mask.unsqueeze(1).unsqueeze(-1).expand(-1, Na, -1, self.d_model)

        # Máscara para la SUMA (Aportan 0.0)
        x_sum = x.masked_fill(mask_exp, 0.0)
        summed_features = torch.sum(x_sum, dim=2)

        # Máscara para el MÁXIMO (Aportan -inf)
        x_max = x.masked_fill(mask_exp, float('-inf'))
        maxed_features, _ = torch.max(x_max, dim=2)
        maxed_features = torch.clamp(maxed_features, min=-10.0, max=10.0)

        # Combinamos Suma + Máximo
        accumulated_features = torch.cat([summed_features, maxed_features], dim=-1)

        # --- STEP 5: PROYECCIÓN DE SALIDA ---
        logits = self.output_projection(accumulated_features).squeeze(-1) # [B, Na]
        logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits
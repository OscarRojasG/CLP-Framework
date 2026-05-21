import torch
import torch.nn as nn
from models.base.mlp_encoder import MLPEncoder
from models.base.transformer import Transformer


class CLPTransformer(Transformer):
    def __init__(self, block_dim, space_dim, d_model=256, nhead=8, num_layers=3, ff_dim_multiplier=4, dropout=0.1):
        # Eliminamos placed_dim del constructor y pasamos space_dim tanto para space como para placed
        super().__init__(
            block_dim=block_dim, 
            space_dim=space_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier,
            dropout=dropout
        )
        self.d_model = d_model

        # 1. Proyecciones base
        self.block_proj = nn.Linear(block_dim, d_model)
        
        # Proyección lineal simple unificada para geometrías (space y placed)
        self.geom_proj = nn.Linear(space_dim, d_model)
        
        # Mantenemos únicamente el encoder de bloques libres
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)

        # 2. Bloque de Self-Attention para las geometrías unificadas
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * ff_dim_multiplier, 
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.self_attention_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # 3. Proyecciones para el Scaled Dot-Product final
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        self.alpha = nn.Parameter(torch.tensor(1.0))

    def encode(self, block_features):
        # Mantenemos el encoder tal como está
        B, N_blocks, _ = block_features.shape

        x = self.block_proj(block_features) # [B, N, d_model]
        x = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)

        return (x, )

    def decode(self, memory, action_blocks, placed_features, space_features, vcs):
        B = memory.shape[0]
        
        # Aseguramos que space_features tenga la dimensión secuencial lista para la concatenación
        if len(space_features.shape) == 2:
            space_features = space_features.unsqueeze(1) # [B, 1, space_dim]

        # --- CONCATENACIÓN PREVIA DE LAS GEOMETRÍAS CRUDAS ---
        # Unificamos el espacio disponible con los bloques ya posicionados en el contenedor
        combined_raw = torch.cat([space_features, placed_features], dim=1) # [B, 1 + N_placed, space_dim]

        # --- PROYECCIÓN LINEAL SIMPLE ---
        # Pasamos todo el conjunto geométrico de manera simultánea por la misma proyección
        combined_features = self.geom_proj(combined_raw) # [B, 1 + N_placed, d_model]

        # --- GESTIÓN DE MÁSCARAS DE PADDING ---
        # Evaluamos las filas válidas (no nulas) en placed_features
        placed_mask = (placed_features.sum(dim=-1) != 0) # [B, N_placed]
        
        # El espacio (index 0) siempre se considera válido
        space_mask = torch.ones((B, 1), dtype=torch.bool, device=placed_features.device)
        combined_mask = torch.cat([space_mask, placed_mask], dim=1) # [B, 1 + N_placed]

        # El Transformer de PyTorch requiere un booleano True en las posiciones que DEBEN ignorarse
        src_key_padding_mask = ~combined_mask

        # --- SELF ATTENTION ---
        # Los elementos geométricos interactúan entre sí para capturar el estado del contenedor
        enriched_features = self.self_attention_block(combined_features, src_key_padding_mask=src_key_padding_mask)

        # --- EXTRACCIÓN DEL ESPACIO ENRIQUECIDO ---
        # Extraemos el vector que corresponde al token del espacio (index 0)
        enriched_space = enriched_features[:, 0:1, :] # [B, 1, d_model]

        # --- EMBEDDINGS DE ACCIONES DESDE MEMORY ---
        action_mask = action_blocks != -1 
        action_idx = action_blocks.clamp(min=0)
        action_emb = torch.gather(memory, 1, action_idx.unsqueeze(-1).expand(-1, -1, self.d_model)) # [B, Na, d_model]

        # --- SCALED DOT PRODUCT ---
        q = self.q_proj(enriched_space) # [B, 1, d_model]
        k = self.k_proj(action_emb)      # [B, Na, d_model]
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)
        transformer_logits = scores.squeeze(1)

        # ---------------------------------------------------
        # NUEVO: INYECCIÓN DE LA HEURÍSTICA VCS COMO BIAS RESIDUAL
        # ---------------------------------------------------
        
        final_logits = transformer_logits + (self.alpha * vcs)

        # Las acciones de relleno deben seguir siendo -inf (¡importante aplicar después del bias!)
        action_mask = action_blocks != -1 
        final_logits = final_logits.masked_fill(~action_mask, float('-inf'))

        return final_logits
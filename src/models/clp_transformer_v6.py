import torch
import torch.nn as nn
from models.base.mlp_encoder import MLPEncoder
from models.base.transformer import Transformer


class CLPTransformer(Transformer):
    def __init__(self, block_dim, action_dim, space_dim, d_model=256, nhead=8, num_layers=3, ff_dim_multiplier=4, dropout=0.1):
        # Registramos las dimensiones correspondientes en la clase base
        super().__init__(
            block_dim=block_dim, 
            action_dim=action_dim,
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
        self.action_proj = nn.Linear(action_dim, d_model)
        self.geom_proj = nn.Linear(space_dim, d_model)
        
        # Encoders basados en MLP
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        self.action_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        
        # Proyección final para fusionar la identidad del bloque (memory) con las features de la acción
        self.final_action_proj = nn.Linear(2 * d_model, d_model)

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

    def encode(self, block_features):
        # El encoder se mantiene intacto
        B, N_blocks, _ = block_features.shape

        x = self.block_proj(block_features) # [B, N, d_model]
        x = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)

        return (x, )

    def decode(self, memory, action_blocks, action_features, placed_features, space_features):
        B = memory.shape[0]
        
        # Aseguramos que space_features tenga la dimensión secuencial lista para la concatenación
        if len(space_features.shape) == 2:
            space_features = space_features.unsqueeze(1) # [B, 1, space_dim]

        # --- CONCATENACIÓN PREVIA DE LAS GEOMETRÍAS CRUDAS ---
        # Unificamos el espacio disponible con los bloques ya posicionados en el contenedor
        combined_raw = torch.cat([space_features, placed_features], dim=1) # [B, 1 + N_placed, space_dim]

        # --- PROYECCIÓN LINEAL SIMPLE ---
        combined_features = self.geom_proj(combined_raw) # [B, 1 + N_placed, d_model]

        # --- GESTIÓN DE MÁSCARAS DE PADDING PARA GEOMETRÍAS ---
        placed_mask = (placed_features.sum(dim=-1) != 0) # [B, N_placed]
        space_mask = torch.ones((B, 1), dtype=torch.bool, device=placed_features.device)
        combined_mask = torch.cat([space_mask, placed_mask], dim=1) # [B, 1 + N_placed]

        src_key_padding_mask = ~combined_mask

        # --- SELF ATTENTION ---
        # Las geometrías interactúan en el espacio del contenedor
        enriched_features = self.self_attention_block(combined_features, src_key_padding_mask=src_key_padding_mask)

        # --- EXTRACCIÓN DEL ESPACIO ENRIQUECIDO (Query) ---
        enriched_space = enriched_features[:, 0:1, :] # [B, 1, d_model]

        # --- EMBEDDINGS DE ACCIONES (Key / Value) ---
        action_mask = action_blocks != -1 
        action_idx = action_blocks.clamp(min=0)
        
        # A. Recuperamos los embeddings base de los bloques desde 'memory'
        block_emb_action = torch.gather(memory, 1, action_idx.unsqueeze(-1).expand(-1, -1, self.d_model)) # [B, Na, d_model]
        
        # B. Procesamos las características dinámicas de la acción con el MLPEncoder
        action_feat_proj = self.action_proj(action_features)
        action_extra = self.action_encoder(action_feat_proj) # [B, Na, d_model]
        
        # C. Concatenamos la identidad del bloque con su contexto de acción (d_model * 2) y proyectamos a d_model
        action_cat = torch.cat([block_emb_action, action_extra], dim=-1) # [B, Na, 2 * d_model]
        action_emb = self.final_action_proj(action_cat) # [B, Na, d_model]

        # --- SCALED DOT PRODUCT ---
        q = self.q_proj(enriched_space) # [B, 1, d_model]
        k = self.k_proj(action_emb)      # [B, Na, d_model]
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)
        logits = scores.squeeze(1) # [B, Na]

        # Enmascaramos las acciones inválidas con -inf
        logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits
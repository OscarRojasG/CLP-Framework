import torch
import torch.nn as nn
from models.base.mlp_encoder import MLPEncoder
from models.base.transformer import Transformer


class CLPTransformer(Transformer):
    def __init__(self, block_dim, space_dim, d_model=256, nhead=8, num_layers=3, ff_dim_multiplier=4, dropout=0.1):
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
        self.geom_proj = nn.Linear(space_dim, d_model)
        
        # Encoder de bloques libres
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)

        # Capa de Cross-Attention eficiente para los 10,000 bloques futuros
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )

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
        B, N_blocks, _ = block_features.shape

        # Extracción de la máscara de bloques válidos (True = Válido, False = Padding de -1)
        block_mask = (block_features != -1).any(dim=-1) # [B, N_blocks]

        # Procesamiento por el MLPEncoder
        x = self.block_proj(block_features) # [B, N_blocks, d_model]
        x = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)

        return x, block_mask

    def decode(self, memory, memory_mask, action_blocks, placed_features, space_features):
        B = memory.shape[0]
        
        # Aseguramos que space_features tenga la dimensión secuencial lista
        if len(space_features.shape) == 2:
            space_features = space_features.unsqueeze(1) # [B, 1, space_dim]

        # 1. CONCATENACIÓN Y PROYECCIÓN DE LAS GEOMETRÍAS CRUDAS
        combined_raw = torch.cat([space_features, placed_features], dim=1) # [B, 1 + N_placed, space_dim]
        geom_features = self.geom_proj(combined_raw) # [B, 1 + N_placed, d_model]

        # 2. GESTIÓN DE MÁSCARAS DE PADDING PARA LAS GEOMETRÍAS
        placed_mask = (placed_features.sum(dim=-1) != 0) # [B, N_placed]
        space_mask = torch.ones((B, 1), dtype=torch.bool, device=placed_features.device)
        combined_mask = torch.cat([space_mask, placed_mask], dim=1) # [B, 1 + N_placed]
        src_key_padding_mask = ~combined_mask

        # --- 3. PASO DE CROSS-ATTENTION (INFORMACIÓN DEL FUTURO EN TIEMPO LINEAL) ---
        # Invertimos la máscara del encoder para cumplir con la API de PyTorch (True = Ignorar)
        cross_padding_mask = ~memory_mask # [B, N_blocks]

        # Query: Estado geométrico actual del contenedor [B, 1 + N_placed, d_model]
        # Key / Value: Embeddings del inventario futuro [B, 10000, d_model]
        context_features, _ = self.cross_attention(
            query=geom_features,
            key=memory,
            value=memory,
            key_padding_mask=cross_padding_mask
        )

        # Suma residual: Inyectamos el inventario futuro filtrado directamente en los tokens geométricos
        geom_features = geom_features + context_features

        # --- 4. SELF-ATTENTION GLOBAL DE GEOMETRÍAS ---
        # Los elementos del contenedor interactúan condicionados por la información de la memoria
        enriched_features = self.self_attention_block(geom_features, src_key_padding_mask=src_key_padding_mask)

        # --- 5. EXTRACCIÓN DEL ESPACIO ENRIQUECIDO ---
        enriched_space = enriched_features[:, 0:1, :] # [B, 1, d_model]

        # --- 6. EMBEDDINGS DE ACCIONES DESDE MEMORY ---
        action_mask = action_blocks != -1 
        action_idx = action_blocks.clamp(min=0)
        action_emb = torch.gather(memory, 1, action_idx.unsqueeze(-1).expand(-1, -1, self.d_model)) # [B, Na, d_model]

        # --- 7. SCALED DOT PRODUCT FINAL ---
        q = self.q_proj(enriched_space) # [B, 1, d_model]
        k = self.k_proj(action_emb)      # [B, Na, d_model]
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)
        logits = scores.squeeze(1) # [B, Na]

        # Enmascaramos las acciones inválidas con -inf
        logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits
import torch
import torch.nn as nn
from models.base.mlp_encoder import MLPEncoder
from models.base.transformer import Transformer


class CLPTransformer(Transformer):
    def __init__(self, block_dim, action_dim, space_dim, d_model=64, nhead=4, num_layers=3, ff_dim_multiplier=3, dropout=0.1):
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
        self.geom_proj = nn.Linear(space_dim+1, d_model)
        
        # Encoders basados en MLP
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        self.action_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        
        # --- MODIFICACIÓN SOLUCIÓN B ---
        # Ahora fusionamos: identidad del bloque (d_model) + features de acción (d_model) + contexto de espacio (d_model)
        self.final_action_proj = nn.Linear(3 * d_model, d_model)

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
        x = self.block_proj(block_features) 
        x = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)
        return (x, )

    def decode(self, memory, action_blocks, action_features, placed_features, space_features, available_blocks):
        B = memory.shape[0]
        Na = action_blocks.shape[1] # Número de acciones candidatas en este paso
        
        # 1. CÁLCULO DEL PERFIL DE INVENTARIO DISPONIBLE
        c_ini = torch.mean(memory, dim=1, keepdim=True) 
        
        avail_mask = available_blocks != -1 
        avail_idx = available_blocks.clamp(min=0)
        avail_embs = torch.gather(memory, 1, avail_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        
        avail_mask_exp = avail_mask.unsqueeze(-1).float()
        inv_sum = (avail_embs * avail_mask_exp).sum(dim=1, keepdim=True)
        inv_count = avail_mask_exp.sum(dim=1, keepdim=True).clamp(min=1)
        c_disp = inv_sum / inv_count 
        
        delta_inventario = c_ini - c_disp 

        # 2. PREPARACIÓN DE GEOMETRÍAS LOCALES CON FLAGS DE IDENTIDAD
        if len(space_features.shape) == 2:
            space_features = space_features.unsqueeze(1) 

        B, N_placed, _ = placed_features.shape
        
        space_flag = torch.ones((B, 1, 1), device=space_features.device)
        placed_flag = torch.zeros((B, N_placed, 1), device=placed_features.device)
        
        space_raw = torch.cat([space_features, space_flag], dim=-1)   
        placed_raw = torch.cat([placed_features, placed_flag], dim=-1) 

        # 3. CONCATENACIÓN TOTAL EN LA SECUENCIA DE AUTO-ATENCIÓN
        space_emb = self.geom_proj(space_raw)
        placed_emb = self.geom_proj(placed_raw)
        
        combined_features = torch.cat([delta_inventario, space_emb, placed_emb], dim=1) 

        # 4. CONFIGURACIÓN DE MÁSCARAS DE PADDING
        is_padding = torch.all(placed_features == -1.0, dim=-1) 
        placed_mask = ~is_padding
        
        inv_mask = torch.ones((B, 1), dtype=torch.bool, device=placed_features.device)
        sp_mask = torch.ones((B, 1), dtype=torch.bool, device=placed_features.device)
        
        combined_mask = torch.cat([inv_mask, sp_mask, placed_mask], dim=1)
        src_key_padding_mask = ~combined_mask

        # 5. AUTO-ATENCIÓN CONTEXTUAL
        enriched_features = self.self_attention_block(combined_features, src_key_padding_mask=src_key_padding_mask)

        # 6. EXTRACCIÓN DEL QUERY (Índice 1: Espacio enriquecido por el entorno e inventario)
        enriched_space = enriched_features[:, 1:2, :] # [B, 1, d_model]

        # 7. PROCESAMIENTO DE ACCIONES CANDIDATO (MODIFICACIÓN SOLUCIÓN B)
        action_mask = action_blocks != -1 
        action_idx = action_blocks.clamp(min=0)
        
        # A. Recuperamos embeddings base de los bloques candidatos
        block_emb_action = torch.gather(memory, 1, action_idx.unsqueeze(-1).expand(-1, -1, self.d_model)) # [B, Na, d_model]
        
        # B. Procesamos las características de la acción (métricas analíticas del entorno)
        action_feat_proj = self.action_proj(action_features)
        action_extra = self.action_encoder(action_feat_proj) # [B, Na, d_model]
        
        # C. INYECCIÓN DEL CONTEXTO ESPACIAL EN CADA ACCIÓN CANDIDATA
        # Expandimos el vector del espacio para que esté presente en cada una de las Na acciones
        space_context_per_action = enriched_space.expand(-1, Na, -1) # [B, Na, d_model]
        
        # Concatenamos los 3 componentes: Identidad del bloque + Pérdida analítica + Espacio físico
        action_cat = torch.cat([block_emb_action, action_extra, space_context_per_action], dim=-1) # [B, Na, 3 * d_model]
        
        # Proyectamos la combinación tridimensional de vuelta a d_model
        action_emb = self.final_action_proj(action_cat) # [B, Na, d_model]

        # 8. SCALED DOT PRODUCT FINAL
        q = self.q_proj(enriched_space) # [B, 1, d_model]
        k = self.k_proj(action_emb)      # [B, Na, d_model]
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)
        logits = scores.squeeze(1) # [B, Na]

        logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits
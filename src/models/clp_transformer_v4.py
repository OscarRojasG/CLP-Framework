import torch
import torch.nn as nn
from models.base.mlp_encoder import MLPEncoder
from abc import ABC, abstractmethod

class Transformer(nn.Module, ABC):
    def __init__(self, **hyperparams):
        torch.manual_seed(42)
        super(Transformer, self).__init__()
        self.hyperparams = hyperparams
        self.biased = False
    
    @abstractmethod
    def encode(self, *args):
        pass

    @abstractmethod
    def decode(self, *args):
        pass

    def forward(self, block_features, box_features, total_boxes, boxes_per_block, *args):
        enc_data = self.encode(block_features, box_features, total_boxes, boxes_per_block)
        return self.decode(*enc_data, *args)


class CLPTransformer(Transformer):
    def __init__(self, box_dim, block_dim, action_dim, placed_dim, d_model=64, nhead=4, num_layers=3, ff_dim_multiplier=3, dropout=0.1):
        # Mantenemos el super por compatibilidad, aunque internamente ignoramos las dimensiones sobrantes
        super().__init__(
            box_dim=box_dim, block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_boxes = 30

        # ======================================================
        # --- 1. RUTA DEL ENCODER (CATÁLOGO / INVENTARIO) ---
        # ======================================================
        self.block_proj = nn.Linear(block_dim, d_model)
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)

        self.box_proj = nn.Linear(box_dim, d_model)
        
        # Pooling de Inventario
        self.inv_query_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.inv_pooling_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.summary_proj = nn.Linear(d_model, d_model)
        self.summary_dropout = nn.Dropout(dropout)
        self.norm_enrich = nn.LayerNorm(d_model)

        # ======================================================
        # --- 2. RUTA DE ACCIONES CANDIDATO (IZQUIERDA) ---
        # ======================================================
        # Proyección para métricas operacionales: [loss, cs] -> Entrada fija de dimensión 2
        self.action_feat_proj = nn.Linear(action_dim, d_model)
        
        # Fusión de contexto global (suma de inventarios)
        self.context_fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model), # Fusiona block_inv y box_inv
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # La fusión de acciones se mantiene, pero ahora recibirá bloques ya "enriquecidos"
        self.query_fusion_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model * ff_dim_multiplier),
            nn.ReLU(),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.Dropout(dropout)
        )

        # --- 3. RUTA DE ENTORNO: CONTENEDOR ENRIQUECIDO ---
        self.placed_proj = nn.Linear(placed_dim, d_model)
        
        # Token único que representa el contenedor global
        self.container_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.empty_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Atención para que el contenedor recolecte información de los bloques
        self.container_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.container_norm = nn.LayerNorm(d_model)

        # --- 4. DECODER: Acciones atienden al Contenedor ---
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.decoder_ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * ff_dim_multiplier),
                nn.ReLU(),
                nn.Linear(d_model * ff_dim_multiplier, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        self.attn_dropout = nn.Dropout(dropout)
        
        self.dec_norm1_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dec_norm2_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.final_dec_norm = nn.LayerNorm(d_model)

        # ======================================================
        # --- 5. CABEZAL DE SALIDA (LOGITS DE ENERGÍA) ---
        # ======================================================
        self.energy_proj = nn.Linear(d_model, 1)

    def encode(self, block_features, box_features, total_boxes, boxes_per_block):
        B, N_blocks, _ = block_features.shape
        
        # 1. Proyección de Bloques
        block_padding_mask = torch.all(block_features == -1.0, dim=-1)
        x = self.block_proj(block_features) # [B, N_blocks, d_model]
    
        # 2. Proyección de Cajas
        box_memory = self.box_proj(box_features) # [B, N_boxes, d_model]
    
        # 3. Operación de agregación local (boxes_per_block)
        box_contribution = torch.matmul(boxes_per_block.float(), box_memory)
        x = x + box_contribution
    
        # --- NUEVO: 4. Operación de agregación global (box_inv) ---
        # Convertimos a 0 los valores de padding (-1) antes de operar
        clean_total = total_boxes.clamp(min=0).float() # [B, N_boxes]
        
        # [B, 1, N_boxes] @ [B, N_boxes, d_model] -> [B, 1, d_model]
        box_inv = torch.matmul(clean_total.unsqueeze(1), box_memory)
    
        # 5. Resto del flujo (Encoder + Pooling)
        block_memory = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)
    
        query = self.inv_query_token.expand(B, -1, -1)
        block_inv, _ = self.inv_pooling_attn(
            query=query, key=block_memory, value=block_memory, key_padding_mask=block_padding_mask
        )
        
        # Retornamos el estado de los bloques enriquecido + el inventario global
        return block_memory, box_memory, block_inv, box_inv, boxes_per_block

    def decode(self, block_memory, box_memory, block_inv, box_inv, boxes_per_block, action_blocks, action_features, placed_blocks, placed_features):
        B = block_memory.shape[0]

        # 2. Identificamos qué bloques están colocados
        placed_mask = (placed_blocks != -1).float() # [B, N_pblocks]
        
        # 3. Recopilamos las cajas de los bloques colocados
        # gather_indices: [B, N_pblocks, N_boxes]
        gather_indices = placed_blocks.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.max_boxes)
        # placed_quantities: [B, N_pblocks, N_boxes]
        placed_quantities = torch.gather(boxes_per_block, 1, gather_indices)
        
        # 4. Sumamos las cantidades (aplicando máscara para ignorar el padding de placed_blocks)
        # total_consumed_per_box: [B, N_boxes]
        total_consumed_per_box = (placed_quantities * placed_mask.unsqueeze(-1)).sum(dim=1)
        
        # 5. Calculamos el embedding de lo consumido y restamos
        # consumed_emb: [B, d_model]
        consumed_emb = torch.matmul(total_consumed_per_box.unsqueeze(1).float(), box_memory)
        box_inv = box_inv - consumed_emb # [B, 1, d_model]

        # Contexto
        global_context = self.context_fusion_mlp(torch.cat([block_inv, box_inv], dim=-1))
        global_context_expanded = global_context.expand(-1, action_blocks.size(1), -1)

        # --- Corrección en el método decode ---
        # 1. Primero calcula los embeddings necesarios
        action_mask = (action_blocks != -1)
        action_idx_clamped = action_blocks.clamp(min=0)
        block_emb_action = torch.gather(block_memory, 1, action_idx_clamped.unsqueeze(-1).expand(-1, -1, self.d_model))
        action_feat_emb = self.action_feat_proj(action_features)
        
        # 2. Ahora sí puedes concatenar los 3 pilares correctamente
        combined_features = torch.cat([block_emb_action, action_feat_emb, global_context_expanded], dim=-1)
        
        # 3. Y finalmente pasarlo por la MLP de fusión
        queries = self.query_fusion_mlp(combined_features)

        # --- 2. RUTA DERECHA: Contenedor Enriquecido ---
        placed_mask = ~torch.all(placed_features == -1.0, dim=-1) # [B, Np]
        placed_sequence = self.placed_proj(placed_features)
        
        # INYECCIÓN: Crear el token de "vacío" y concatenarlo al inicio
        empty_t = self.empty_token.expand(B, 1, -1)
        placed_sequence = torch.cat([empty_t, placed_sequence], dim=1)
        
        # Actualizar máscara: el nuevo token en la pos 0 debe ser False (no ignorar)
        full_mask = torch.cat([torch.zeros(B, 1, device=placed_mask.device, dtype=torch.bool), ~placed_mask], dim=1)
        
        # El contenedor (token único) atiende a todos los bloques (incluyendo el empty_token)
        container = self.container_token.expand(B, -1, -1)
        container_enriched, _ = self.container_attn(
            query=container, 
            key=placed_sequence, 
            value=placed_sequence, 
            key_padding_mask=full_mask
        )
        container_enriched = self.container_norm(container + container_enriched)

        # --- 3. CROSS-ATTENTION: Acciones atienden al Contenedor ---
        for i in range(self.num_layers):
            normed_queries = self.dec_norm1_layers[i](queries)
            # Acciones (Q) atienden al contenedor (K, V)
            attn_output, _ = self.cross_attn_layers[i](
                query=normed_queries, 
                key=container_enriched, 
                value=container_enriched
            )
            queries = queries + self.attn_dropout(attn_output)

            normed_queries_2 = self.dec_norm2_layers[i](queries)
            queries = queries + self.decoder_ff_layers[i](normed_queries_2)

        logits = self.energy_proj(self.final_dec_norm(queries)).squeeze(-1)
        return logits.masked_fill(~action_mask, -1e9)
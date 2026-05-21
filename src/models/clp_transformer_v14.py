import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.num_layers = num_layers

        # 1. Proyecciones base
        self.block_proj = nn.Linear(block_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.geom_proj = nn.Linear(space_dim+1, d_model)
        
        # Encoders basados en MLP
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        self.action_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        
        # --- ATENCIÓN DE INVENTARIO BIDIRECCIONAL (ENCODER) ---
        self.inv_query_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.inv_pooling_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        
        self.summary_proj = nn.Linear(d_model, d_model)
        self.summary_dropout = nn.Dropout(dropout)
        self.norm_enrich = nn.LayerNorm(d_model)

        # --- TOKEN VACÍO DE SEGURIDAD ANTI-NaNs ---
        self.empty_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- FUSIÓN DE ACCIÓN CANDIDATA ---
        self.final_action_proj = nn.Linear(2 * d_model, d_model)

        # --- 2. SECUENCIA PROFUNDA DE CROSS-ATTENTION (ESPACIO -> BLOQUES COLOCADOS) ---
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.attn_dropout = nn.Dropout(dropout)
        
        self.decoder_ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ff_dim_multiplier * d_model),
                nn.ReLU(),
                nn.Linear(ff_dim_multiplier * d_model, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.dec_norm1_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dec_norm2_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # --- 3. NUEVA CAPA DE DECISIÓN FINAL (CROSS ATTENTION NORMAL + PROYECCIÓN) 🚨 ---
        # Las acciones (Queries) buscan compatibilidad en el espacio geométrico consolidado (Keys/Values)
        self.final_cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.final_attn_dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        
        # Proyección de salida a 1 solo valor final por acción
        self.output_projection = nn.Linear(d_model, 1)

    def encode(self, block_features):
        B, N_blocks, _ = block_features.shape
        block_padding_mask = torch.all(block_features == -1.0, dim=-1)
        
        x = self.block_proj(block_features) 
        raw_memory = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)
        
        query = self.inv_query_token.expand(B, -1, -1)
        inv_summary, _ = self.inv_pooling_attn(
            query=query,
            key=raw_memory,
            value=raw_memory,
            key_padding_mask=block_padding_mask
        )
        
        contextual_modifier = self.summary_proj(inv_summary)
        contextual_modifier = self.summary_dropout(contextual_modifier)
        
        enriched_memory = self.norm_enrich(raw_memory + contextual_modifier)
        return enriched_memory, 

    def decode(self, memory, action_blocks, action_features, placed_features, space_features):
        B = memory.shape[0]
        Na = action_blocks.shape[1] 

        if len(space_features.shape) == 2:
            space_features = space_features.unsqueeze(1) 

        B, N_placed, _ = placed_features.shape
        
        space_flag = torch.ones((B, 1, 1), device=space_features.device)
        placed_flag = torch.zeros((B, N_placed, 1), device=placed_features.device)
        
        space_raw = torch.cat([space_features, space_flag], dim=-1)   
        placed_raw = torch.cat([placed_features, placed_flag], dim=-1) 

        space_emb = self.geom_proj(space_raw)   
        placed_emb = self.geom_proj(placed_raw) 

        dummy_key = self.empty_token.expand(B, -1, -1)
        keys = torch.cat([dummy_key, placed_emb], dim=1)
        values = keys 

        is_padding = torch.all(placed_features == -1.0, dim=-1) 
        placed_mask = ~is_padding
        dummy_mask = torch.zeros((B, 1), dtype=torch.bool, device=placed_features.device)
        key_padding_mask = torch.cat([dummy_mask, ~placed_mask], dim=1) 

        # 3. LOOP DE CROSS-ATTENTION CONTEXTUAL PROFUNDO
        enriched_space = space_emb
        for i in range(self.num_layers):
            attn_output, _ = self.cross_attn_layers[i](
                query=enriched_space, 
                key=keys,
                value=values,
                key_padding_mask=key_padding_mask
            )
            enriched_space = self.dec_norm1_layers[i](enriched_space + self.attn_dropout(attn_output))
            
            ff_output = self.decoder_ff_layers[i](enriched_space)
            enriched_space = self.dec_norm2_layers[i](enriched_space + ff_output)

        # 4. PROCESAMIENTO DE ACCIONES CANDIDATO
        action_mask = action_blocks != -1 
        action_idx = action_blocks.clamp(min=0)
        
        block_emb_action = torch.gather(memory, 1, action_idx.unsqueeze(-1).expand(-1, -1, self.d_model)) 
        action_feat_proj = self.action_proj(action_features)
        action_extra = self.action_encoder(action_feat_proj) 
        
        action_cat = torch.cat([block_emb_action, action_extra], dim=-1) 
        action_emb = self.final_action_proj(action_cat) # [B, Na, d_model]

        # --- 5. NUEVA SELECCIÓN MEDIANTE CROSS ATTENTION NORMAL 🚨 ---
        # Query: Las acciones candidatas [B, Na, d_model]
        # Key / Value: El espacio enriquecido tras escanear el contenedor [B, 1, d_model]
        attn_out, _ = self.final_cross_attn(
            query=action_emb,
            key=enriched_space,
            value=enriched_space
        ) # Output shape: [B, Na, d_model]
        
        # Conexión residual y normalización estándar de Transformer
        action_emb = self.final_norm(action_emb + self.final_attn_dropout(attn_out))

        # Proyección lineal final a un único valor por acción
        logits = self.output_projection(action_emb).squeeze(-1) # [B, Na]

        # Enmascaramiento rígido final
        logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits
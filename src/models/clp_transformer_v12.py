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
        self.inv_back_proj_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm_enrich = nn.LayerNorm(d_model)

        # --- TOKEN VACÍO DE SEGURIDAD ANTI-NaNs ---
        self.empty_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- FUSIÓN DE ACCIÓN CANDIDATA ---
        self.final_action_proj = nn.Linear(3 * d_model, d_model)

        # --- 2. SECUENCIA PROFUNDA DE CROSS-ATTENTION (ESPACIO -> BLOQUES COLOCADOS) 🚨 ---
        # Reemplazamos la capa única por listas de módulos apilados de num_layers capas
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
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

        # 3. Proyecciones para el Scaled Dot-Product final
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

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
        
        contextual_blocks, _ = self.inv_back_proj_attn(
            query=raw_memory,
            key=inv_summary,
            value=inv_summary
        )
        
        enriched_memory = self.norm_enrich(raw_memory + contextual_blocks)
        return enriched_memory, 

    def decode(self, memory, action_blocks, action_features, placed_features, space_features):
        B = memory.shape[0]
        Na = action_blocks.shape[1] 

        # 1. PREPARACIÓN DE GEOMETRÍAS LOCALES CON FLAGS DE IDENTIDAD
        if len(space_features.shape) == 2:
            space_features = space_features.unsqueeze(1) 

        B, N_placed, _ = placed_features.shape
        
        space_flag = torch.ones((B, 1, 1), device=space_features.device)
        placed_flag = torch.zeros((B, N_placed, 1), device=placed_features.device)
        
        space_raw = torch.cat([space_features, space_flag], dim=-1)   
        placed_raw = torch.cat([placed_features, placed_flag], dim=-1) 

        space_emb = self.geom_proj(space_raw)   # [B, 1, d_model]
        placed_emb = self.geom_proj(placed_raw) # [B, N_placed, d_model]

        # 2. CONFIGURACIÓN DE LAS KEYS CON EL EMPTY TOKEN DE SEGURIDAD
        dummy_key = self.empty_token.expand(B, -1, -1)
        keys = torch.cat([dummy_key, placed_emb], dim=1)
        values = keys 

        is_padding = torch.all(placed_features == -1.0, dim=-1) 
        placed_mask = ~is_padding
        dummy_mask = torch.zeros((B, 1), dtype=torch.bool, device=placed_features.device)
        key_padding_mask = torch.cat([dummy_mask, ~placed_mask], dim=1) 

        # 3. LOOP DE CROSS-ATTENTION CONTEXTUAL PROFUNDO (MHA + FF SEQUENTIAL) 🚨 ---
        # Enriched space actúa como la Query iterativa que absorbe no-linealidades de los bloques colocados
        enriched_space = space_emb
        for i in range(self.num_layers):
            # Subcapa 1: Multi-Head Cross-Attention + Conexión Residual + LayerNorm
            attn_output, _ = self.cross_attn_layers[i](
                query=enriched_space, 
                key=keys,
                value=values,
                key_padding_mask=key_padding_mask
            )
            enriched_space = self.dec_norm1_layers[i](enriched_space + attn_output)
            
            # Subcapa 2: Feed Forward Network + Conexión Residual + LayerNorm
            ff_output = self.decoder_ff_layers[i](enriched_space)
            enriched_space = self.dec_norm2_layers[i](enriched_space + ff_output)

        # 4. PROCESAMIENTO DE ACCIONES CANDIDATO
        action_mask = action_blocks != -1 
        action_idx = action_blocks.clamp(min=0)
        
        block_emb_action = torch.gather(memory, 1, action_idx.unsqueeze(-1).expand(-1, -1, self.d_model)) 
        
        action_feat_proj = self.action_proj(action_features)
        action_extra = self.action_encoder(action_feat_proj) 
        
        space_context_per_action = enriched_space.expand(-1, Na, -1) 
        
        action_cat = torch.cat([block_emb_action, action_extra, space_context_per_action], dim=-1) 
        action_emb = self.final_action_proj(action_cat) 

        # 5. SCALED DOT PRODUCT FINAL
        q = self.q_proj(enriched_space) 
        k = self.k_proj(action_emb)      
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)
        logits = scores.squeeze(1) 

        logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits
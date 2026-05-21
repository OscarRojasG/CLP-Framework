import torch
import torch.nn as nn
from models.base.mlp_encoder import MLPEncoder
from models.base.transformer import Transformer

class CLPTransformer(Transformer):
    def __init__(self, block_dim, action_dim, placed_dim, d_model=64, nhead=4, num_layers=3, ff_dim_multiplier=3, dropout=0.1):
        # Mantenemos el super por compatibilidad, aunque internamente ignoramos las dimensiones sobrantes
        super().__init__(
            block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        self.d_model = d_model
        self.num_layers = num_layers

        # ======================================================
        # --- 1. RUTA DEL ENCODER (CATÁLOGO / INVENTARIO) ---
        # ======================================================
        self.block_proj = nn.Linear(block_dim, d_model)
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)
        
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
        
        # MLP de Fusión para Query definitiva: Concatena block_emb (d_model) + action_feat_emb (d_model)
        self.query_fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model * ff_dim_multiplier),
            nn.ReLU(),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.Dropout(dropout)
        )

        # ======================================================
        # --- 3. RUTA DE ENTORNO TOPOLÓGICO (DERECHA) ---
        # ======================================================
        # Proyección directa de la geometría relativa: Prisma local -> Entrada fija de dimensión 6
        self.placed_proj = nn.Linear(placed_dim, d_model)
        
        # Self-Attention en Pre-LN para modelar co-ocurrencia de obstáculos
        self.placed_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.placed_norm1 = nn.LayerNorm(d_model)
        self.placed_dropout = nn.Dropout(dropout)
        
        # Token vacío de seguridad anti-NaNs para el contenedor vacío
        self.empty_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ======================================================
        # --- 4. DECODER DE CROSS-ATTENTION SUPERIOR ---
        # ======================================================
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

        # --- NUEVO: Autocalibración competitiva de acciones candidatos ---
        self.action_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.action_self_norm = nn.LayerNorm(d_model)
        self.action_self_dropout = nn.Dropout(dropout)

    def encode(self, block_features):
        B, N_blocks, _ = block_features.shape
        block_padding_mask = torch.all(block_features == -1.0, dim=-1)

        x = self.block_proj(block_features)
        raw_memory = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)

        query = self.inv_query_token.expand(B, -1, -1)
        inv_summary, _ = self.inv_pooling_attn(
            query=query, key=raw_memory, value=raw_memory, key_padding_mask=block_padding_mask
        )
        contextual_modifier = self.summary_dropout(self.summary_proj(inv_summary))
        enriched_memory = self.norm_enrich(raw_memory + contextual_modifier)
        return enriched_memory,

    def decode(self, memory, action_blocks, placed_features, action_features):
        B = memory.shape[0]
        Na = action_blocks.shape[1]

        # --- 1. RUTA IZQUIERDA: QUERIES (Candidatos informados por memoria) ---
        action_mask = (action_blocks != -1)
        action_idx_clamped = action_blocks.clamp(min=0)

        # Extraemos el embedding del catálogo [B, Na, d_model]
        block_emb_action = torch.gather(memory, 1, action_idx_clamped.unsqueeze(-1).expand(-1, -1, self.d_model))
        
        # Proyectamos las métricas operacionales (loss, cs)
        action_feat_emb = self.action_feat_proj(action_features)   # nn.Linear(2, d_model)
        
        # Fusión inicial
        query_concat = torch.cat([block_emb_action, action_feat_emb], dim=-1)
        queries = self.query_fusion_mlp(query_concat) # [B, Na, d_model]

        # 🚨 MEJORA DE ESTABILIZACIÓN: Self-Attention Competitivo entre las 64 acciones
        # Forzamos a que los logits de las acciones se calibren mutuamente antes de mirar el entorno
        normed_queries = self.action_self_norm(queries)
        
        # Generamos la máscara de padding de acciones para que no miren canales vacíos (-inf)
        # MultiheadAttention de PyTorch usa True para elementos que deben ignorarse
        action_padding_mask = ~action_mask 
        
        queries_calibrated, _ = self.action_self_attn(
            query=normed_queries, key=normed_queries, value=normed_queries, 
            key_padding_mask=action_padding_mask
        )
        # Conexión residual: las queries ahora están informadas de todo el conjunto competitivo
        queries = queries + self.action_self_dropout(queries_calibrated)

        # --- 2. RUTA DERECHA: KEYS/VALUES (Topología Relativa Pura) ---
        # Identificamos el padding del entorno basándonos en si las coordenadas son -1.0
        placed_mask = ~torch.all(placed_features == -1.0, dim=-1)

        # Proyección directa de la geometría relativa (las 6 variables del prisma local)
        # Aquí eliminamos la concatenación semántica vieja. El entorno es 100% geométrico.
        placed_sequence = self.placed_proj(placed_features)       # nn.Linear(6, d_model)

        # Máscara anti-ruido para estabilizar LayerNorm
        mask_expanded = placed_mask.unsqueeze(-1).expand(-1, -1, self.d_model)
        placed_sequence = placed_sequence * mask_expanded

        # Inyección de token vacío de seguridad y máscaras de padding extendidas
        dummy_placed_token = self.empty_token.expand(B, -1, -1)
        unified_placed_sequence = torch.cat([dummy_placed_token, placed_sequence], dim=1)
        
        dummy_mask = torch.zeros((B, 1), dtype=torch.bool, device=placed_features.device)
        extended_placed_mask = torch.cat([dummy_mask, ~placed_mask], dim=1)

        # Self-Attention en estructura Pre-LN Estricta
        normed_placed = self.placed_norm1(unified_placed_sequence)
        attn_output, _ = self.placed_self_attn(
            query=normed_placed, key=normed_placed, value=normed_placed, key_padding_mask=extended_placed_mask
        )
        keys_values = unified_placed_sequence + self.placed_dropout(attn_output)

        # --- 3. CROSS-ATTENTION SUPERIOR ---
        for i in range(self.num_layers):
            normed_queries = self.dec_norm1_layers[i](queries)
            attn_output, _ = self.cross_attn_layers[i](
                query=normed_queries, key=keys_values, value=keys_values, key_padding_mask=extended_placed_mask
            )
            queries = queries + self.attn_dropout(attn_output)

            normed_queries_2 = self.dec_norm2_layers[i](queries)
            ff_output = self.decoder_ff_layers[i](normed_queries_2)
            queries = queries + ff_output

        queries = self.final_dec_norm(queries)

        # --- 4. EXTRACCIÓN DE LOGITS ---
        energy_scores = self.energy_proj(queries)
        logits = energy_scores.squeeze(-1)
        logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits
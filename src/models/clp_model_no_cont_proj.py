import torch
import torch.nn as nn
from models.base.transformer import Transformer
from models.clp_model import build_mlp

# Variante de CLPModelNoCont con la misma simplificación que CLPModelProj: la atención sobre una sola key
# no depende de la query, así que se reemplaza por algo equivalente. Como aquí la key es un vector constante,
# W_c t_const + b_c es a su vez un vector constante, y cada capa de scoring se reduce a sumar un vector
# aprendido (d parámetros por capa) a todas las acciones.
#
# CLPModelNoCont es una ablación de CLPModel sin container stream:
#   - Se eliminan la proyección de bloques colocados, el token vacío y las capas cont_layers.
#   - Las capas de scoring son idénticas, pero usan como key/value un vector constante aprendido
#     (el mismo para todos los estados) en vez del contenedor enriquecido.
#   - placed_dim y placed_features se mantienen solo por compatibilidad con la interfaz de datos; no se usan.

class CLPModelNoContProj(Transformer):
    def __init__(self, block_dim, action_dim, placed_dim, d_model=64, nhead=4, scoring_layers=3,
                 enc_layers=2, ff_dim_multiplier=3, dropout=0.1):
        super().__init__(
            block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim,
            d_model=d_model, nhead=nhead, scoring_layers=scoring_layers,
            enc_layers=enc_layers, ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        self.d_model = d_model
        self.scoring_layers = scoring_layers

        # ======================================================
        # --- 1. ENCODER MLP (CATÁLOGO / INVENTARIO) ---
        # ======================================================
        self.block_mlp = build_mlp(block_dim, d_model * ff_dim_multiplier, d_model, enc_layers, dropout)

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

        # --- 3-4. DECODER: Un vector constante aprendido por capa, sumado a las acciones ---
        # (equivale a W_c t_const + b_c; se inicializa en cero, como el sesgo de nn.Linear)
        self.const_layers = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, d_model)) for _ in range(scoring_layers)
        ])
        self.decoder_ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * ff_dim_multiplier),
                nn.ReLU(),
                nn.Linear(d_model * ff_dim_multiplier, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(scoring_layers)
        ])
        self.attn_dropout = nn.Dropout(dropout)

        self.dec_norm2_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(scoring_layers)])
        self.final_dec_norm = nn.LayerNorm(d_model)

        # ======================================================
        # --- 5. CABEZAL DE SALIDA (LOGITS DE ENERGÍA) ---
        # ======================================================
        self.energy_proj = nn.Linear(d_model, 1)

    def encode(self, block_features):
        # Cada bloque se codifica de forma independiente, por lo que el padding
        # no contamina a los bloques válidos (el decoder solo lee índices válidos)
        return self.block_mlp(block_features),

    def decode(self, memory, action_blocks, action_features, placed_features):
        # --- 1. RUTA IZQUIERDA: Preparar Acciones (Sin Self-Attention) ---
        action_mask = (action_blocks != -1)
        action_idx_clamped = action_blocks.clamp(min=0)
        block_emb_action = torch.gather(memory, 1, action_idx_clamped.unsqueeze(-1).expand(-1, -1, self.d_model))
        action_feat_emb = self.action_feat_proj(action_features)

        # queries base [B, Na, d_model]
        queries = self.query_fusion_mlp(torch.cat([block_emb_action, action_feat_emb], dim=-1))

        # --- 2. SCORING: Vector constante de cada capa sumado a todas las acciones (sin información del contenedor) ---
        for i in range(self.scoring_layers):
            queries = queries + self.attn_dropout(self.const_layers[i])

            normed_queries_2 = self.dec_norm2_layers[i](queries)
            queries = queries + self.decoder_ff_layers[i](normed_queries_2)

        logits = self.energy_proj(self.final_dec_norm(queries)).squeeze(-1)
        return logits.masked_fill(~action_mask, -1e9)

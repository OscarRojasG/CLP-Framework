import torch
import torch.nn as nn
from models.base.transformer import Transformer
from models.clp_model import build_mlp
from models.clp_model_factored import build_single_key_attention, single_key_attention, convert_attention_state_dict

# Variante de CLPModelNoCont equivalente al 100% (salida, inicialización, dropout y optimización), sin los
# parámetros que no reciben gradiente. Aplica la misma factorización que CLPModelFactored: con una sola key
# (el vector constante), la atención de cada capa de scoring es W_o (W_v t_const + b_v) + b_o, y las
# proyecciones Q y K y dec_norm1_layers se eliminan.
#
# CLPModelNoCont es una ablación de CLPModel sin container stream:
#   - Se eliminan la proyección de bloques colocados, el token vacío y las capas cont_layers.
#   - Las capas de scoring son idénticas, pero usan como key/value un vector constante aprendido
#     (el mismo para todos los estados) en vez del contenedor enriquecido.
#   - placed_dim y placed_features se mantienen solo por compatibilidad con la interfaz de datos; no se usan.

class CLPModelNoContFactored(Transformer):
    def __init__(self, block_dim, action_dim, placed_dim, d_model=64, nhead=4, scoring_layers=3,
                 enc_layers=2, ff_dim_multiplier=3, dropout=0.1, seed=None):
        super().__init__(
            seed=seed,
            block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim,
            d_model=d_model, nhead=nhead, scoring_layers=scoring_layers,
            enc_layers=enc_layers, ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        self.d_model = d_model
        self.nhead = nhead
        self.scoring_layers = scoring_layers
        self.dropout = dropout

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

        # --- 3. VECTOR CONSTANTE (reemplaza al contenedor enriquecido) ---
        self.const_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- 4. DECODER: W_o (W_v t_const + b_v) + b_o sumado a las acciones ---
        self.value_proj_layers = nn.ModuleList()
        self.out_proj_layers = nn.ModuleList()
        for _ in range(scoring_layers):
            value_proj, out_proj = build_single_key_attention(d_model, nhead, dropout)
            self.value_proj_layers.append(value_proj)
            self.out_proj_layers.append(out_proj)

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
        B = memory.shape[0]

        # --- 1. RUTA IZQUIERDA: Preparar Acciones (Sin Self-Attention) ---
        action_mask = (action_blocks != -1)
        action_idx_clamped = action_blocks.clamp(min=0)
        block_emb_action = torch.gather(memory, 1, action_idx_clamped.unsqueeze(-1).expand(-1, -1, self.d_model))
        action_feat_emb = self.action_feat_proj(action_features)

        # queries base [B, Na, d_model]
        queries = self.query_fusion_mlp(torch.cat([block_emb_action, action_feat_emb], dim=-1))

        # --- 2. Vector constante como key/value (sin información del contenedor) ---
        const = self.const_token.expand(B, -1, -1)

        # --- 3. SCORING: Contribución del vector constante sumada a las acciones ---
        for i in range(self.scoring_layers):
            attn_output = single_key_attention(
                self.value_proj_layers[i], self.out_proj_layers[i], const,
                queries.shape[1], self.nhead, self.dropout, self.training
            )
            queries = queries + self.attn_dropout(attn_output)

            normed_queries_2 = self.dec_norm2_layers[i](queries)
            queries = queries + self.decoder_ff_layers[i](normed_queries_2)

        logits = self.energy_proj(self.final_dec_norm(queries)).squeeze(-1)
        return logits.masked_fill(~action_mask, -1e9)

    @staticmethod
    def convert_state_dict(clp_state_dict, d_model):
        """Convierte el state_dict de un CLPModelNoCont (ver convert_attention_state_dict)."""
        return convert_attention_state_dict(clp_state_dict, d_model)

    @staticmethod
    def convert_proj_state_dict(proj_state_dict, d_model):
        """
        Convierte el state_dict de un CLPModelNoContProj (un vector constante c_i por capa): W_v = 0, b_v = 0,
        W_o = I, b_o = c_i y t_const = 0. La salida es idéntica en inferencia.
        """
        converted = {}
        for key, value in proj_state_dict.items():
            if key.startswith("const_layers."):
                i = key.split(".")[1]
                converted[f"value_proj_layers.{i}.weight"] = torch.zeros(d_model, d_model, dtype=value.dtype)
                converted[f"value_proj_layers.{i}.bias"] = torch.zeros(d_model, dtype=value.dtype)
                converted[f"out_proj_layers.{i}.weight"] = torch.eye(d_model, dtype=value.dtype)
                converted[f"out_proj_layers.{i}.bias"] = value.reshape(d_model)
                continue
            converted[key] = value
        converted["const_token"] = torch.zeros(1, 1, d_model)
        return converted

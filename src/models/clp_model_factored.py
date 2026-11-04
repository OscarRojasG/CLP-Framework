import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.transformer import Transformer
from models.clp_model import build_mlp

# Variante de CLPModel equivalente al 100% (salida, inicialización, dropout y optimización), sin los
# parámetros que no reciben gradiente.
#
# En las capas de scoring de CLPModel la atención tiene una sola key (el contenedor), por lo que el
# softmax siempre vale 1 y la salida es W_o (W_v t_c + b_v) + b_o, independiente de la query.
# Las proyecciones Q y K y dec_norm1_layers nunca reciben gradiente, así que se eliminan.
#
# Para que el entrenamiento sea idéntico (y no solo la familia de funciones):
#   - Se mantienen los dos factores W_v y W_o por separado. Fusionarlos en una sola matriz (como hacía
#     CLPModelProj) cambia la dinámica de AdamW y del weight decay.
#   - Se replica el dropout que nn.MultiheadAttention aplica al peso de atención (que vale 1): anula o
#     reescala la contribución de cada cabeza, de forma independiente por acción.
#   - La inicialización consume el generador aleatorio en el mismo orden que CLPModel, por lo que con
#     la misma semilla se obtienen los mismos pesos.


def build_single_key_attention(d_model, nhead, dropout):
    """
    Proyecciones V y de salida de una nn.MultiheadAttention, sin Q ni K.
    Se construye una atención temporal (consumiendo el generador aleatorio igual que nn.MultiheadAttention)
    y se copian sus proyecciones, por lo que con la misma semilla se obtienen los mismos pesos.
    """
    attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
    value_proj = nn.utils.skip_init(nn.Linear, d_model, d_model)
    out_proj = nn.utils.skip_init(nn.Linear, d_model, d_model)
    with torch.no_grad():
        value_proj.weight.copy_(attn.in_proj_weight[2 * d_model:])
        value_proj.bias.copy_(attn.in_proj_bias[2 * d_model:])
        out_proj.weight.copy_(attn.out_proj.weight)
        out_proj.bias.copy_(attn.out_proj.bias)
    return value_proj, out_proj


def single_key_attention(value_proj, out_proj, key_value, num_queries, nhead, dropout, training):
    """
    Salida de nn.MultiheadAttention (batch_first) con una sola key: W_o (W_v kv + b_v) + b_o para cada query.
    key_value: [B, 1, d_model]. Devuelve [B, num_queries, d_model].

    En entrenamiento replica el dropout que nn.MultiheadAttention aplica al peso de atención (que vale 1),
    que anula o reescala de forma independiente cada cabeza de cada query.
    """
    B, _, d_model = key_value.shape
    values = value_proj(key_value)  # [B, 1, d_model]

    if training and dropout > 0:
        head_dim = d_model // nhead
        # Mismo tensor (y mismo consumo del generador aleatorio) que el peso de atención de MultiheadAttention
        head_weights = F.dropout(values.new_ones(B * nhead, num_queries, 1), p=dropout)
        head_weights = head_weights.view(B, nhead, num_queries, 1)
        values = values.view(B, 1, nhead, head_dim).transpose(1, 2)  # [B, nhead, 1, head_dim]
        values = (head_weights * values).permute(2, 0, 1, 3).reshape(num_queries, B, d_model)

        # Igual que MultiheadAttention con batch_first: la salida es una vista transpuesta de un tensor
        # [num_queries, B, d_model]. El dropout posterior genera la máscara en el orden de la memoria, por lo que
        # se mantiene esta disposición para obtener las mismas máscaras.
        return out_proj(values).transpose(0, 1)

    return out_proj(values.expand(B, num_queries, d_model))


def convert_attention_state_dict(clp_state_dict, d_model):
    """
    Convierte el state_dict de CLPModel o CLPModelNoCont al de su variante factorizada: extrae W_v y b_v
    de in_proj y descarta las proyecciones Q y K y dec_norm1_layers (que no reciben gradiente).
    """
    converted = {}
    for key, value in clp_state_dict.items():
        if key.startswith("dec_norm1_layers."):
            continue
        if key.startswith("cross_attn_layers."):
            _, i, name = key.split(".", 2)
            if name == "in_proj_weight":
                converted[f"value_proj_layers.{i}.weight"] = value[2 * d_model:]
            elif name == "in_proj_bias":
                converted[f"value_proj_layers.{i}.bias"] = value[2 * d_model:]
            elif name == "out_proj.weight":
                converted[f"out_proj_layers.{i}.weight"] = value
            elif name == "out_proj.bias":
                converted[f"out_proj_layers.{i}.bias"] = value
            continue
        converted[key] = value
    return converted


class CLPModelFactored(Transformer):
    """
    El encoder de bloques es una MLP clásica aplicada a cada bloque por separado.
    En el decoder, cada capa suma a las acciones candidato W_o (W_v t_c + b_v) + b_o, con t_c el contenedor enriquecido.
    """
    def __init__(self, block_dim, action_dim, placed_dim, d_model=64, nhead=4, scoring_layers=3,
                 cont_layers=2, enc_layers=2, ff_dim_multiplier=3, dropout=0.1, seed=None):
        super().__init__(
            seed=seed,
            block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim,
            d_model=d_model, nhead=nhead, scoring_layers=scoring_layers,
            cont_layers=cont_layers, enc_layers=enc_layers,
            ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        self.d_model = d_model
        self.nhead = nhead
        self.scoring_layers = scoring_layers
        self.cont_layers = cont_layers
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

        # --- 3. RUTA DE ENTORNO: CONTENEDOR ENRIQUECIDO ---
        self.placed_proj = nn.Linear(placed_dim, d_model)

        # Token único que representa el contenedor global
        self.container_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.empty_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Capas apiladas para que el contenedor recolecte información de los bloques
        self.container_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
            for _ in range(cont_layers)
        ])
        self.container_ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * ff_dim_multiplier),
                nn.ReLU(),
                nn.Linear(d_model * ff_dim_multiplier, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(cont_layers)
        ])
        self.cont_norm1_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(cont_layers)])
        self.cont_norm2_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(cont_layers)])
        # Normalización final del contenedor antes de usarlo como memoria del decoder
        self.container_norm = nn.LayerNorm(d_model)

        # --- 4. DECODER: W_o (W_v t_c + b_v) + b_o sumado a las acciones ---
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
        for i in range(self.cont_layers):
            container_attn_out, _ = self.container_attn_layers[i](
                query=self.cont_norm1_layers[i](container),
                key=placed_sequence,
                value=placed_sequence,
                key_padding_mask=full_mask
            )
            container = container + self.attn_dropout(container_attn_out)
            container = container + self.container_ff_layers[i](self.cont_norm2_layers[i](container))

        container_enriched = self.container_norm(container)

        # --- 3. SCORING: Contribución del contenedor sumada a las acciones ---
        for i in range(self.scoring_layers):
            attn_output = single_key_attention(
                self.value_proj_layers[i], self.out_proj_layers[i], container_enriched,
                queries.shape[1], self.nhead, self.dropout, self.training
            )
            queries = queries + self.attn_dropout(attn_output)

            normed_queries_2 = self.dec_norm2_layers[i](queries)
            queries = queries + self.decoder_ff_layers[i](normed_queries_2)

        logits = self.energy_proj(self.final_dec_norm(queries)).squeeze(-1)
        return logits.masked_fill(~action_mask, -1e9)

    @staticmethod
    def convert_state_dict(clp_state_dict, d_model):
        """Convierte el state_dict de un CLPModel (ver convert_attention_state_dict)."""
        return convert_attention_state_dict(clp_state_dict, d_model)

    @staticmethod
    def convert_proj_state_dict(proj_state_dict, d_model):
        """
        Convierte el state_dict de un CLPModelProj (W_c t_c + b_c en una sola matriz): W_v = W_c, b_v = b_c,
        W_o = I y b_o = 0. La salida es idéntica en inferencia.
        """
        converted = {}
        for key, value in proj_state_dict.items():
            if key.startswith("container_proj_layers."):
                _, i, name = key.split(".", 2)
                converted[f"value_proj_layers.{i}.{name}"] = value
                if name == "weight":
                    converted[f"out_proj_layers.{i}.weight"] = torch.eye(d_model, dtype=value.dtype)
                    converted[f"out_proj_layers.{i}.bias"] = torch.zeros(d_model, dtype=value.dtype)
                continue
            converted[key] = value
        return converted

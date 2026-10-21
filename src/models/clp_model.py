import torch
import torch.nn as nn
from models.base.transformer import Transformer

# Variante de CLPMLPFix (Pre-LN) con los siguientes cambios:
#   - El encoder de bloques es una MLP clásica de enc_layers capas (sin normalización ni conexión residual).
#   - El contenedor se enriquece con cont_layers capas apiladas (cross-attention + FF, Pre-LN),
#     siguiendo el mismo flujo que la atención entre acciones y contenedor (antes una sola atención).

def build_mlp(in_dim, hidden_dim, out_dim, num_layers, dropout):
    """
    MLP clásica (sin normalización ni conexión residual) con num_layers capas lineales.
    Con num_layers=1 equivale a una proyección lineal.
    """
    assert num_layers >= 1, "La MLP debe tener al menos una capa"
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class CLPModel(Transformer):
    """
    El encoder de bloques es una MLP clásica aplicada a cada bloque por separado.
    El decoder hace que las acciones candidato atiendan a un contenedor enriquecido.
    """
    def __init__(self, block_dim, action_dim, placed_dim, d_model=64, nhead=4, scoring_layers=3,
                 cont_layers=2, enc_layers=2, ff_dim_multiplier=3, dropout=0.1):
        super().__init__(
            block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim,
            d_model=d_model, nhead=nhead, scoring_layers=scoring_layers,
            cont_layers=cont_layers, enc_layers=enc_layers,
            ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        self.d_model = d_model
        self.scoring_layers = scoring_layers
        self.cont_layers = cont_layers

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

        # --- 4. DECODER: Acciones atienden al Contenedor ---
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
            for _ in range(scoring_layers)
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

        self.dec_norm1_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(scoring_layers)])
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

        # --- 3. CROSS-ATTENTION: Acciones atienden al Contenedor ---
        for i in range(self.scoring_layers):
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

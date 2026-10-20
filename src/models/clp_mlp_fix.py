import torch.nn as nn
from models.clp_transformer_fix import CLPTransformerFix

class MLPBlock(nn.Module):
    def __init__(self, d_model, ff_dim_multiplier, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_dim_multiplier),
            nn.ReLU(),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-LN: x = x + sublayer(norm(x))
        return x + self.mlp(self.norm(x))

class CLPMLPFix(CLPTransformerFix):
    """
    Variante de CLPTransformerFix (Pre-LN) donde el encoder de bloques es una MLP simple
    aplicada a cada bloque por separado (sin InventoryAttention).
    El decoder es el mismo de CLPTransformerFix.
    """
    def __init__(self, block_dim, action_dim, placed_dim, d_model=64, nhead=4, num_layers=3,
                 ff_dim_multiplier=3, dropout=0.1, encoder_layers=2):
        super().__init__(
            block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        # Registrado para que load_model pueda reconstruir el modelo
        self.hyperparams["encoder_layers"] = encoder_layers

        # Eliminamos la ruta de InventoryAttention heredada
        del self.inventory_layers
        del self.inv_query_token

        # ======================================================
        # --- 1. ENCODER MLP (CATÁLOGO / INVENTARIO) ---
        # ======================================================
        # self.block_proj (Linear inicial) y self.final_enc_norm se heredan de CLPTransformerFix
        self.encoder_layers = nn.Sequential(*[
            MLPBlock(d_model, ff_dim_multiplier, dropout)
            for _ in range(encoder_layers)
        ])

    def encode(self, block_features):
        # Cada bloque se codifica de forma independiente, por lo que el padding
        # no contamina a los bloques válidos (el decoder solo lee índices válidos)
        x_blocks = self.block_proj(block_features)
        x_blocks = self.encoder_layers(x_blocks)

        return self.final_enc_norm(x_blocks),

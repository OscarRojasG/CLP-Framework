import torch.nn as nn
from models.clp_transformer import CLPTransformer

class MLPBlock(nn.Module):
    def __init__(self, d_model, ff_dim_multiplier, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_dim_multiplier),
            nn.ReLU(),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.mlp(x))

class CLPMLP(CLPTransformer):
    """
    Variante de CLPTransformer donde el encoder de bloques es una MLP simple
    aplicada a cada bloque por separado (sin InventoryAttention).
    El decoder es el mismo de CLPTransformer.
    """
    def __init__(self, block_dim, action_dim, placed_dim, d_model=64, nhead=4, encoder_layers=2, decoder_layers=3,
                 ff_dim_multiplier=3, dropout=0.1):
        super().__init__(
            block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim,
            d_model=d_model, nhead=nhead, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
            ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        # Eliminamos la ruta de InventoryAttention heredada
        del self.inventory_layers
        del self.inv_query_token

        # ======================================================
        # --- 1. ENCODER MLP (CATÁLOGO / INVENTARIO) ---
        # ======================================================
        # self.block_proj (Linear inicial) se hereda de CLPTransformer
        self.encoder_layers = nn.Sequential(*[
            MLPBlock(d_model, ff_dim_multiplier, dropout)
            for _ in range(encoder_layers)
        ])

    def encode(self, block_features):
        # Cada bloque se codifica de forma independiente, por lo que el padding
        # no contamina a los bloques válidos (el decoder solo lee índices válidos)
        x_blocks = self.block_proj(block_features)
        x_blocks = self.encoder_layers(x_blocks)

        return x_blocks,

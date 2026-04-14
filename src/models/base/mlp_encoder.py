import torch.nn as nn

class MLPBlock(nn.Module):
    """Bloque individual con conexión residual y normalización"""
    def __init__(self, d_model, ff_dim_multiplier, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_dim_multiplier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.ffn(x))

class MLPEncoder(nn.Module):
    def __init__(self, d_model, ff_dim_multiplier, dropout, num_layers=1):
        super().__init__()
        self.layers = nn.Sequential(*[
            MLPBlock(d_model, ff_dim_multiplier, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, x):
        return self.layers(x)
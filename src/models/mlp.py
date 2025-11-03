import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout_rate=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),  # no uses LayerNorm aquí
            ]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        # Capa final de salida
        self.output = nn.Linear(in_dim, 1)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = x.float()
        h = self.mlp(x)
        out = self.output(h).squeeze(-1)
        return out
import torch
import torch.nn as nn

class FeatureExpansion(nn.Module):
    """Expande cada feature en tres representaciones: x, log(x), log(1-x)."""
    def forward(self, x):
        # Reescalar a (0,1) con una sigmoide
        x_scaled = torch.sigmoid(x)
        eps = 1e-8
        x_clamped = x_scaled.clamp(eps, 1 - eps)
        return torch.cat([
            x_scaled,
            torch.log(x_clamped),
            torch.log(1 - x_clamped)
        ], dim=-1)


class AdaptativeDecoderModel(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim, num_layers=2, dropout_rate=0):
        super(AdaptativeDecoderModel, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # ✅ Bloque de expansión de features
        self.feature_expansion = FeatureExpansion()
        expanded_dim = input_dim * 3  # x, log(x), log(1-x)

        # Proyección de entrada
        self.input_projection = nn.Linear(expanded_dim, num_heads * head_dim)

        # Capas de atención + densa
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'multihead_attention': nn.MultiheadAttention(
                    embed_dim=num_heads * head_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate
                ),
                'dense_layer': nn.Sequential(
                    nn.Linear(num_heads * head_dim, 512),
                    nn.Tanh(),
                    nn.Linear(512, num_heads * head_dim)
                ),
                'norm1': nn.LayerNorm(num_heads * head_dim),
                'norm2': nn.LayerNorm(num_heads * head_dim)
            }) for _ in range(num_layers)
        ])

        # Proyección de salida
        self.output_projection = nn.Linear(num_heads * head_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]
        x = x.float()

        # ✅ Expansión de características
        x_expanded = self.feature_expansion(x)

        # Proyección de entrada
        x_proj = self.input_projection(x_expanded)

        # Aplicar capas de atención + dense
        for layer in self.layers:
            x_proj = x_proj.permute(1, 0, 2)
            attn_output, _ = layer['multihead_attention'](x_proj, x_proj, x_proj)
            attn_output = attn_output.permute(1, 0, 2)
            x_proj = x_proj.permute(1, 0, 2)
            attn_output = layer['norm1'](attn_output + x_proj)
            dense_output = layer['dense_layer'](attn_output)
            x_proj = layer['norm2'](dense_output + attn_output)

        # Proyección de salida final
        output = self.output_projection(x_proj)  # [batch, seq_len, 1]

        # Flatten a [batch_size, seq_length]
        flat_output = output.view(output.size(0), -1)
        return flat_output

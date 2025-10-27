import torch.nn as nn
import torch.nn.functional as F

class DecoderModel(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim, num_layers=2, dropout_rate=0):
        super(DecoderModel, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Proyección de entrada
        self.input_projection = nn.Linear(input_dim, num_heads * head_dim)

        # Crear múltiples capas de atención y densa
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'multihead_attention': nn.MultiheadAttention(
                    embed_dim=num_heads * head_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate
                ),
                'dense_layer': nn.Sequential(
                    nn.Linear(num_heads * head_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_heads * head_dim)
                ),
                'norm1': nn.LayerNorm(num_heads * head_dim),
                'norm2': nn.LayerNorm(num_heads * head_dim)
            }) for _ in range(num_layers)
        ])

        # Proyección de salida final
        self.output_projection = nn.Linear(num_heads * head_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]
        x = x.float()

        # Proyección de entrada
        x_proj = self.input_projection(x)

        # Aplicar cada capa de atención y densa
        for layer in self.layers:
            x_proj = x_proj.permute(1, 0, 2)
            attn_output, _ = layer['multihead_attention'](
                x_proj, x_proj, x_proj
            )
            attn_output = attn_output.permute(1, 0, 2)
            x_proj = x_proj.permute(1, 0, 2)
            attn_output = layer['norm1'](attn_output + x_proj)
            dense_output = layer['dense_layer'](attn_output)
            x_proj = layer['norm2'](dense_output + attn_output)

        # Proyección de salida final (logits)
        output = self.output_projection(x_proj)  # [batch_size, seq_length, num_classes]

        # Flatten
        flat_output = output.view(output.size(0), -1)  # [batch_size, seq_length]

        return flat_output  # Retorna los logits (sin aplicar softmax)
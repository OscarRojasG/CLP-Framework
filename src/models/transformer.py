import torch.nn as nn
from .base.encoder_decoder import EncoderDecoderModel

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        """
        x: [batch, src_len, input_dim]
        return: [batch, src_len, embed_dim]
        """
        return self.net(x)


class Transformer(EncoderDecoderModel):
    def __init__(self, src_dim, tgt_dim, num_heads, head_dim, num_layers=2, dropout_rate=0):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        embed_dim = num_heads * head_dim

        # --- Encoder reemplazado por MLP ---
        self.encoder = MLPEncoder(src_dim, embed_dim)

        # --- Decoder ---
        self.decoder_input_proj = nn.Linear(tgt_dim, embed_dim)
        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True),
                'cross_attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, embed_dim)
                ),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'norm3': nn.LayerNorm(embed_dim)
            }) for _ in range(num_layers)
        ])

        # --- Output layer ---
        self.output_projection = nn.Linear(embed_dim, 1)

    def forward(self, src, tgt):
        """
        src: [batch, src_len, src_dim] -> bloques fijos
        tgt: [batch, tgt_len, tgt_dim] -> secuencia parcial de acciones
        """

        # --- Encoder con MLP ---
        memory = self.encoder(src)  # [batch, src_len, embed_dim]

        # --- Decoder ---
        tgt_proj = self.decoder_input_proj(tgt)
        for layer in self.decoder_layers:
            # self-attention en decoder
            self_attn_out, _ = layer['self_attn'](tgt_proj, tgt_proj, tgt_proj)
            tgt_proj = layer['norm1'](tgt_proj + self_attn_out)

            # cross-attention con encoder
            cross_attn_out, _ = layer['cross_attn'](tgt_proj, memory, memory)
            tgt_proj = layer['norm2'](tgt_proj + cross_attn_out)

            # feed-forward
            ffn_out = layer['ffn'](tgt_proj)
            tgt_proj = layer['norm3'](tgt_proj + ffn_out)

        # --- Salida ---
        output = self.output_projection(tgt_proj)  # [batch, tgt_len, 1]
        # Flatten
        flat_output = output.view(output.size(0), -1) # [batch, tgt_len]
        return flat_output # Retorna los logits (sin aplicar softmax)
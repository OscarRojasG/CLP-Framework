import torch
import torch.nn as nn
from .base.encoder_decoder import EncoderDecoderModel

class ChunkedTransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers=2, chunk_size=512, dropout_rate=0):
        super().__init__()
        self.chunk_size = chunk_size

        # Proyección inicial de características
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Capas tipo Transformer encoder (reutilizables por chunk)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, embed_dim)
                ),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim)
            })
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: [batch, src_len, input_dim]
        Devuelve: [batch, src_len, embed_dim]
        """
        B, L, _ = x.shape
        x = self.input_proj(x)

        # Dividir en chunks
        chunks = x.split(self.chunk_size, dim=1)
        encoded_chunks = []

        for chunk in chunks:
            out = chunk
            for layer in self.layers:
                # Autoatención dentro del chunk
                attn_out, _ = layer['self_attn'](out, out, out)
                out = layer['norm1'](out + attn_out)

                # Feed-forward
                ffn_out = layer['ffn'](out)
                out = layer['norm2'](out + ffn_out)
            encoded_chunks.append(out)

        # Concatenar de nuevo
        return torch.cat(encoded_chunks, dim=1)
    
class Transformer(EncoderDecoderModel):
    def __init__(self, src_dim, tgt_dim, num_heads, head_dim, num_layers=2, dropout_rate=0):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        embed_dim = num_heads * head_dim

        # --- Encoder reemplazado por MLP ---
        self.encoder = ChunkedTransformerEncoder(
            input_dim=src_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            chunk_size=512,
            dropout_rate=dropout_rate
        )

        # --- Decoder ---
        self.decoder_input_proj = nn.Linear(tgt_dim, embed_dim)
        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True),
                'cross_attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, embed_dim)
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
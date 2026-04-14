import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_dim_multiplier, dropout):
        super().__init__()
        
        # 1. Bloque de Atención
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Bloque Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim_multiplier * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim_multiplier * d_model, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None):
        # --- Rama de Atención ---
        attn_out, _ = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )
        # Dropout residual y suma antes de la norma (Post-LN)
        x = self.norm1(query + self.dropout1(attn_out))
        
        # --- Rama Feed-Forward ---
        ff_out = self.ff(x)
        # Dropout residual y suma antes de la norma (Post-LN)
        x = self.norm2(x + self.dropout2(ff_out))
        
        return x
    
class CrossAttentionDecoder(nn.Module):
    def __init__(self, d_model, nhead, ff_dim_multiplier, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, ff_dim_multiplier, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Refina el 'query' (espacio) a través de múltiples capas de atención
        hacia la 'memory' (bloques colocados).
        """
        x = query
        for layer in self.layers:
            x = layer(
                query=x, 
                key=key, 
                value=value, 
                key_padding_mask=key_padding_mask
            )
        return x
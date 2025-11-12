import torch
import torch.nn as nn
import math
from .base.encoder_decoder_pe import EncoderDecoderPEModel

# ======================================================
# --- Sinusoidal 3D Positional Encoding ---
# ======================================================
class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, max_freq=10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
        self.d_per_coord = d_model // 3

    def forward(self, coords):
        """
        coords: (num_blocks, 3)
        Devuelve: (num_blocks, d_model)
        """
        coords = coords / torch.tensor([587.0, 233.0, 220.0], device=coords.device)
        device = coords.device
        n = coords.shape[0]

        div_term = torch.exp(
            torch.arange(0, self.d_per_coord, 2, device=device) *
            (-math.log(self.max_freq) / self.d_per_coord)
        )

        def pe_1d(coord):
            # coord: (N,)
            sin = torch.sin(coord.unsqueeze(1) * div_term)
            cos = torch.cos(coord.unsqueeze(1) * div_term)
            return torch.cat([sin, cos], dim=1)

        pe_x = pe_1d(coords[:, 0])
        pe_y = pe_1d(coords[:, 1])
        pe_z = pe_1d(coords[:, 2])
        pe = torch.cat([pe_x, pe_y, pe_z], dim=1)

        # Ajuste final: truncar o rellenar
        if pe.shape[1] < self.d_model:
            pad = self.d_model - pe.shape[1]
            pe = torch.cat([pe, torch.zeros(n, pad, device=device)], dim=1)
        elif pe.shape[1] > self.d_model:
            pe = pe[:, :self.d_model]

        return pe

# ======================================================
# --- Encoder MLP para X_src (bloques estáticos) ---
# ======================================================

class BlockEncoder(nn.Module):
    def __init__(self, src_dim, d_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(src_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, X_src):
        """
        X_src: (num_blocks, src_dim)
        -> (num_blocks, d_model)
        """
        return self.encoder(X_src)

# ======================================================
# --- MLP para features de acción ---
# ======================================================

class ActionEncoder(nn.Module):
    def __init__(self, action_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, X_action):
        """
        X_action: (num_actions, action_dim)
        -> (num_actions, d_model)
        """
        return self.net(X_action)
    
# ======================================================
# --- Resumen información global encoder ---
# ======================================================
    
class GlobalAggregator(nn.Module):
    def __init__(self, d_model, mode="mean"):
        super().__init__()
        self.mode = mode
        if mode == "attention":
            self.query = nn.Parameter(torch.randn(d_model))
            self.proj = nn.Linear(d_model, 1)

    def forward(self, E_src):
        """
        E_src: (B, N_blocks, d_model)
        Devuelve: (B, d_model)
        """
        if self.mode == "mean":
            return E_src.mean(dim=1)
        elif self.mode == "max":
            return E_src.max(dim=1).values
        elif self.mode == "attention":
            attn = torch.softmax(self.proj(E_src), dim=1)  # (B, N, 1)
            return (E_src * attn).sum(dim=1)

# ======================================================
# --- Modelo completo con cross-attention ---
# ======================================================

class TransformerPE(EncoderDecoderPEModel):
    def __init__(self, src_dim, tgt_dim, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Componentes principales
        self.block_encoder = BlockEncoder(src_dim, d_model)
        self.action_encoder = ActionEncoder(tgt_dim - 1, d_model)
        self.global_agg = GlobalAggregator(d_model, mode="mean")
        self.pe = PositionalEncoding3D(d_model)

        # Decoder Transformer
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Proyección final a logit
        self.output_head = nn.Linear(d_model, 1)


    def forward(self, X_src, X_tgt, placed, coords):
        """
        X_src:   (B, N_blocks, src_dim)
        X_tgt:   (B, N_actions, 1 + action_feats)
        placed:  (B, min_actions) con índices de bloques colocados (-1 = padding)
        coords:  (B, min_actions, 3) coordenadas relativas
        """
        device = X_src.device
        B, N_blocks, _ = X_src.shape

        # --- Encoder ---
        E_src = self.block_encoder(X_src.view(-1, X_src.shape[-1]))  # (B*N_blocks, d_model)
        E_src = E_src.view(B, N_blocks, self.d_model)

        # --- Contexto con placed + coords ---
        batch_contexts = []
        for b in range(B):
            valid_mask = (placed[b] != -1)
            if valid_mask.any():
                # seleccionar embeddings de los bloques colocados
                ctx = E_src[b, placed[b, valid_mask].long()]  # (N_context, d_model)
                # aplicar PE 3D a las coords válidas
                ctx = ctx + self.pe(coords[b, valid_mask])
            else:
                ctx = torch.zeros((1, self.d_model), device=device)
            batch_contexts.append(ctx)

        # pad las secuencias de contexto para formar un batch rectangular
        max_ctx = max(c.shape[0] for c in batch_contexts)
        E_context = torch.zeros((B, max_ctx, self.d_model), device=device)
        src_key_padding_mask = torch.ones((B, max_ctx), dtype=torch.bool, device=device)
        for b, ctx in enumerate(batch_contexts):
            n = ctx.shape[0]
            E_context[b, :n] = ctx
            src_key_padding_mask[b, :n] = False  # False = posición válida

        # --- Decoder ---
        block_indices = X_tgt[:, :, 0].long()
        X_action = X_tgt[:, :, 1:]

        # obtener embeddings del bloque (índices por batch)
        B_idx = torch.arange(B, device=device).unsqueeze(-1)
        E_block = E_src[B_idx, block_indices]  # (B, N_actions, d_model)
        E_action = self.action_encoder(X_action.view(-1, X_action.shape[-1])).view(B, -1, self.d_model)

        # fusión simple
        E_global = self.global_agg(E_src)  # (B, d_model)
        E_tgt = E_block + E_action + E_global.unsqueeze(1)

        # --- Cross-attention ---
        decoder_out = self.decoder(
            E_tgt, E_context,
            memory_key_padding_mask=src_key_padding_mask
        )  # (B, N_actions, d_model)

        # --- Salida ---
        logits = self.output_head(decoder_out).squeeze(-1)  # (B, N_actions)
        return logits
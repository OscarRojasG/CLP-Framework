import torch
import torch.nn as nn
import math
from .base.encoder_decoder_pe import EncoderDecoderPEModel

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

class Transformer(EncoderDecoderPEModel):
    def __init__(self, src_dim, tgt_dim, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Componentes principales
        self.block_encoder = BlockEncoder(src_dim, d_model)
        self.action_encoder = ActionEncoder(tgt_dim - 1, d_model)
        self.global_agg = GlobalAggregator(d_model, mode="mean")

        # Decoder Transformer
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Proyección bloque + acción (d_model*2) a d_model
        self.action_block_proj = nn.Linear(self.d_model*2, self.d_model)

        # proyectar concatenación memory + coords a d_model
        self.coord_proj = nn.Linear(self.d_model + 3, self.d_model)

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
                ctx = E_src[b, placed[b, valid_mask].long()]  # (N_context, d_model)
                coord_vals = coords[b, valid_mask]
                ctx = torch.cat([ctx, coord_vals], dim=1)  # concatenar coords
            else:
                ctx = torch.zeros((1, self.d_model + 3), device=device)
            batch_contexts.append(ctx)

        # pad secuencias
        max_ctx = max(c.shape[0] for c in batch_contexts)
        E_context = torch.zeros((B, max_ctx, self.d_model + 3), device=device)
        src_key_padding_mask = torch.ones((B, max_ctx), dtype=torch.bool, device=device)
        for b, ctx in enumerate(batch_contexts):
            n = ctx.shape[0]
            E_context[b, :n] = ctx
            src_key_padding_mask[b, :n] = False

        # proyectar concatenación a d_model
        E_context = self.coord_proj(E_context)

        # --- Decoder ---
        block_indices = X_tgt[:, :, 0].long()
        X_action = X_tgt[:, :, 1:]

        # obtener embeddings del bloque (índices por batch)
        B_idx = torch.arange(B, device=device).unsqueeze(-1)
        E_block = E_src[B_idx, block_indices]  # (B, N_actions, d_model)
        E_action = self.action_encoder(X_action.view(-1, X_action.shape[-1])).view(B, -1, self.d_model)

        # concatenar embedding bloque + acción y proyectar a d_model
        E_tgt = torch.cat([E_block, E_action], dim=2)
        E_tgt = self.action_block_proj(E_tgt)

        # sumar embedding global
        E_global = self.global_agg(E_src)  # (B, d_model)
        E_tgt = E_tgt + E_global.unsqueeze(1)

        # --- Cross-attention ---
        decoder_out = self.decoder(
            E_tgt, E_context,
            memory_key_padding_mask=src_key_padding_mask
        )  # (B, N_actions, d_model)

        # --- Salida ---
        logits = self.output_head(decoder_out).squeeze(-1)  # (B, N_actions)
        return logits
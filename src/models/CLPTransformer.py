import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.transformer import Transformer

# ======================================================
# --- Encoder MLP para X_src (bloques estáticos) ---
# ======================================================

class BlockEncoder(nn.Module):
    def __init__(self, block_dim, d_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(block_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, X_src):
        """
        X_src: (num_blocks, block_dim)
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

class CLPTransformer(Transformer):
    def __init__(self, block_dim, action_dim, placed_dim, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super().__init__(
            block_dim=block_dim,
            action_dim=action_dim,
            placed_dim=placed_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        self.d_model = d_model

        # Componentes principales
        self.block_encoder = BlockEncoder(block_dim, d_model)
        self.action_encoder = ActionEncoder(action_dim, d_model)
        self.global_agg = GlobalAggregator(d_model, mode="mean")

        # Decoder Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Proyección bloque + acción → d_model
        self.action_block_proj = nn.Linear(d_model * 2, d_model)

        # Proyección placed_data → embedding d_model
        self.placed_data_proj = nn.Linear(placed_dim, d_model)

        # Proyección contexto concatenado (d_model + d_model) → d_model
        self.placed_block_proj = nn.Linear(d_model * 2, d_model)

        # Proyección final
        self.output_head = nn.Linear(d_model, 1)


    def encode(self, block_features):
        B, N_blocks, _ = block_features.shape

        # Encoder MLP
        E_src = self.block_encoder(
            block_features.view(-1, block_features.shape[-1])
        ).view(B, N_blocks, self.d_model)

        return E_src
    

    def decode(self, memory, action_blocks, action_features, placed_blocks, placed_features):
        B, _, _ = memory.shape
        device = memory.device

        # ----------------------
        # ----- CONTEXTO -------
        # ----------------------
        batch_contexts = []

        for b in range(B):
            valid_mask = (placed_blocks[b] != -1)

            if valid_mask.any():
                # Embedding del bloque colocado
                ctx_block = memory[b, placed_blocks[b][valid_mask]]    # (Nc, d_model)
                
                # Datos adicionales del placed
                placed_data = placed_features[b, valid_mask, :]            # (Nc, placed_dim)
                placed_emb = self.placed_data_proj(placed_data)    # (Nc, d_model)

                # Concatenación → (Nc, 2*d_model)
                ctx = torch.cat([ctx_block, placed_emb], dim=1)
            else:
                # Secuencia vacía → un dummy para que no explote el decoder
                ctx = torch.zeros((1, self.d_model * 2), device=device)

            batch_contexts.append(ctx)

        # PAD
        max_ctx = max(c.shape[0] for c in batch_contexts)

        E_context = torch.zeros((B, max_ctx, self.d_model * 2), device=device)
        src_key_padding_mask = torch.ones((B, max_ctx), dtype=torch.bool, device=device)

        for b, ctx in enumerate(batch_contexts):
            n = ctx.shape[0]
            E_context[b, :n] = ctx
            src_key_padding_mask[b, :n] = False

        # Reducir a d_model
        E_context = self.placed_block_proj(E_context)

        # ----------------------
        # --------- DECODER ----
        # ----------------------
        B_idx = torch.arange(B, device=device).unsqueeze(-1)

        E_block = memory[B_idx, action_blocks]       # (B, N_actions, d_model)
        E_action = self.action_encoder(
            action_features.reshape(-1, action_features.shape[-1])
        ).reshape(B, -1, self.d_model)

        # Concatenación bloque + acción → proyección
        E_tgt = torch.cat([E_block, E_action], dim=2)
        E_tgt = self.action_block_proj(E_tgt)

        # Agregar embedding global
        E_global = self.global_agg(memory).unsqueeze(1)
        E_tgt = E_tgt + E_global

        # ----------------------
        # --- CROSS-ATTENTION --
        # ----------------------
        decoder_out = self.decoder(
            E_tgt,
            E_context,
            memory_key_padding_mask=src_key_padding_mask
        )

        # ----------------------
        # ---- OUTPUT LOGITS ---
        # ----------------------
        logits = self.output_head(decoder_out).squeeze(-1)
        probs = F.softmax(logits, dim=-1)

        return probs
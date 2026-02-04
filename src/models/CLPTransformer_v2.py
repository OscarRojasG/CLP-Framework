import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base.transformer import Transformer

class CLPTransformer(Transformer):
    def __init__(self, blocks_input_dim, space_input_dim, placed_input_dim, embed_dim=128, num_heads=8, num_encoder_layers=2, num_glimpses=2, dropout_rate=0.1):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_glimpses=num_glimpses,
            dropout_rate=dropout_rate,
        )
        self.embed_dim = embed_dim
        
        # --- ENCODER ---
        # Proyección inicial
        self.encoder_input_layer = nn.Linear(blocks_input_dim, embed_dim)
        
        mlp_layers = []
        mlp_layers.append(nn.Linear(embed_dim, embed_dim))

        for _ in range(num_encoder_layers - 1):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(embed_dim, embed_dim))

        self.encoder = nn.Sequential(*mlp_layers)
  
        # --- DECODER ---
        # Proyección placed_data y space
        self.placed_data_proj = nn.Linear(placed_input_dim, embed_dim)
        self.space_proj = nn.Linear(space_input_dim, embed_dim)

        # Fusión de contexto (bloque colocado + data)
        self.ctx_fusion = nn.Linear(2 * embed_dim, embed_dim)

        # Fusión de estado (contexto + espacio + global)
        self.state_fusion = nn.Linear(3 * embed_dim, embed_dim)

        # Cross Attention
        self.num_glimpses = num_glimpses
        self.glimpse_proj = nn.Linear(embed_dim, embed_dim) # Proyección antes de glimpse

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Pointer Scorer
        self.pointer_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    # =====================================================
    # 1. --- ENCODER ---
    # Se llama una sola vez por instancia
    # =====================================================
    def encode(self, x_src):
        """
        x_src: (batch, num_blocks, input_dim) -> Bloques
        """
        # Proyectamos los bloques y los pasamos por el encoder
        enc_input = self.encoder_input_layer(x_src)
        memory = self.encoder(enc_input)  # (batch, num_blocks, embed_dim)

        return memory
    
    # =====================================================
    # 2. --- DECODER ---
    # Se llama en cada paso del rollout
    # =====================================================
    def decode(self, memory, best_blocks, space, placed, placed_data):
        """
        memory: (batch, num_blocks, embed_dim)      -> Salida del encoder
        best_blocks (batch, W)                      -> Índice mejores bloques (-1 para padding)
        space:  (batch, space_input_dim)            -> Data del espacio actual
        placed: (batch, T)                          -> Índice bloques colocados (-1 para padding)
        placed_data: (batch, T, placed_input_dim)   -> Data de los bloques colocados
        """
        B, num_blocks, _ = memory.shape
        _, num_actions = best_blocks.shape
        device = memory.device

        # 2. --- MÁSCARA BLOQUES COLOCADOS ---
        placed_blocks_mask = torch.zeros(
            B, num_blocks, dtype=torch.bool, device=device
        )

        valid = placed != -1
        batch_ids, pos_ids = valid.nonzero(as_tuple=True)

        placed_blocks_mask[batch_ids, placed[batch_ids, pos_ids]] = True

        # 3. --- PROYECTAR COORDENADAS Y ESPACIO ---
        placed_data_emb = self.placed_data_proj(placed_data)   # (B, T, D)
        space_emb = self.space_proj(space)      # (B, D)

        # 4. --- FUSIONAR BLOQUES COLOCADOS + COORDENADAS ---
        # Inicializamos con ceros (para padding)
        placed_blocks_emb = torch.zeros(
            B, placed.size(1), memory.size(-1),
            device=device
        )

        # Tomamos los embeddings correctos desde memory
        placed_blocks_emb[batch_ids, pos_ids] = memory[
            batch_ids, placed[batch_ids, pos_ids]
        ]

        # Concatenar por la dimensión de features
        ctx_emb = torch.cat(
            [placed_blocks_emb, placed_data_emb],
            dim=-1
        )   # (B, T, 2D)

        # Proyectar de vuelta a D
        ctx_emb = self.ctx_fusion(ctx_emb)  # (B, T, D)

        # 5. --- MEDIA DEL CONTEXTO ---
        # Máscara de pasos válidos
        valid_mask = (placed != -1)            # (B, T)
        valid_mask = valid_mask.unsqueeze(-1)  # (B, T, 1)

        # Suma solo de pasos válidos
        ctx_sum = (ctx_emb * valid_mask).sum(dim=1)  # (B, D)

        # Número de pasos válidos por batch
        counts = valid_mask.sum(dim=1).clamp(min=1)  # (B, 1)

        # Media temporal
        ctx_mean = ctx_sum / counts                  # (B, D)

        # 6. --- CONCATENAR Y FUSIONAR CON ESPACIO Y CONTEXTO GLOBAL ---
        global_ctx = memory.mean(dim=1)

        state_emb = torch.cat(
            [ctx_mean, space_emb, global_ctx],
            dim=-1
        )   # (B, 3D)

        # Proyectar de vuelta a D
        state_emb = self.state_fusion(state_emb)  # (B, D)
        
        # 7. --- MÁSCARA W MEJORES BLOQUES ---
        best_blocks_mask = torch.zeros(
            B, num_blocks, dtype=torch.bool, device=device
        )

        valid = best_blocks != -1
        batch_ids, pos_ids = valid.nonzero(as_tuple=True)

        best_blocks_mask[batch_ids, best_blocks[batch_ids, pos_ids]] = True

        # 8. --- DECODER: Cross-Attention (Glimpse) ---
        query = self.glimpse_proj(state_emb).unsqueeze(1)

        for _ in range(self.num_glimpses):
            attn_out, _ = self.cross_attn(
                query=query,            # (B, 1, D)
                key=memory,             # (B, N, D)
                value=memory,           # (B, N, D)
                key_padding_mask=~best_blocks_mask  # (B, N)
            )

            query = self.norm1(attn_out + query)   # Residual + Norm
            ff_out = self.ff(query)                # Feed-Forward
            query = self.norm2(ff_out + query)  # Residual + Norm

        attn_out = query.squeeze(1)         # (B, D)

        # 9. --- DECODER: Pointer scoring ---
        # Máscara de acciones válidas
        valid_actions_mask = (best_blocks != -1)        # (B, K)

        # Inicializar embeddings
        best_blocks_emb = torch.zeros(
            B, num_actions, self.embed_dim,
            device=device
        )

        # Índices válidos
        batch_ids, action_ids = valid_actions_mask.nonzero(as_tuple=True)

        best_blocks_emb[batch_ids, action_ids] = memory[
            batch_ids, best_blocks[batch_ids, action_ids]
        ]  # (B, K, D)

        ptr_query = self.pointer_proj(attn_out)          # (B, D)

        scores = torch.matmul(
            best_blocks_emb,                             # (B, K, D)
            ptr_query.unsqueeze(-1)                      # (B, D, 1)
        ).squeeze(-1)                                    # (B, K)

        scores = scores / math.sqrt(self.embed_dim)

        # 10. --- DECODER: Masking y Softmax ---
        scores = scores.masked_fill(~valid_actions_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)                # (B, K)

        return probs
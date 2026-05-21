import torch
import torch.nn as nn
from models.base.mlp_encoder import MLPEncoder
from models.base.transformer import Transformer


class ValuePredictorTransformer(Transformer):
    def __init__(self, block_dim, action_dim, space_dim, d_model=64, nhead=4, num_layers=3, ff_dim_multiplier=3, dropout=0.1):
        # Mantenemos el super() intacto para respetar la herencia de la firma global
        super().__init__(
            block_dim=block_dim, 
            action_dim=action_dim,
            space_dim=space_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier,
            dropout=dropout
        )
        self.d_model = d_model

        # 1. Proyecciones base de características (Limpias de variables de acción)
        self.block_proj = nn.Linear(block_dim, d_model)
        self.geom_proj = nn.Linear(space_dim + 1, d_model)  # Bloques colocados + flag

        # Encoder basado en MLP únicamente para el catálogo/inventario de bloques
        self.block_encoder = MLPEncoder(d_model, ff_dim_multiplier, dropout, num_layers)

        # Token estructural de Estado Global [CLS]
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 2. Bloque 1: Auto-Atención para esculpir la geometría construida del contenedor
        # Secuencia pequeña: [CLS] + N_placed
        self.container_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * ff_dim_multiplier,
                dropout=dropout, activation='relu', batch_first=True
            ),
            num_layers=num_layers
        )

        # 3. Bloque 2: MULTI-HEAD CROSS-ATTENTION ASIMÉTRICA
        # El contenedor interroga con resolución total al catálogo de inventario de hasta 10.000 piezas
        self.inventory_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # MLP de Regresión Robusto para estimar el volumen final escalar
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def encode(self, block_features):
        B, N_blocks, _ = block_features.shape
        x = self.block_proj(block_features) 
        x = self.block_encoder(x.view(-1, self.d_model)).view(B, N_blocks, self.d_model)
        return (x, )

    def decode(self, memory, action_blocks, action_features, placed_features, space_features, available_blocks):
        B = memory.shape[0]
        device = placed_features.device
        
        # --- 1. EXTRACCIÓN Y MÁSCARA DEL INVENTARIO COMPLETO (KEYS / VALUES) ---
        avail_mask = available_blocks != -1 # [B, N_avail]
        avail_idx = available_blocks.clamp(min=0)
        inventory_embs = torch.gather(memory, 1, avail_idx.unsqueeze(-1).expand(-1, -1, self.d_model)) # [B, N_avail, d_model]
        
        # True = Ignorar en el estándar de PyTorch
        inventory_padding_mask = ~avail_mask # [B, N_avail]

        # --- 2. CONSTRUCCIÓN Y AUTO-ATENCIÓN DE LA ESTRUCTURA FÍSICA (QUERIES) ---
        B, N_placed, _ = placed_features.shape
        placed_flag = torch.zeros((B, N_placed, 1), device=device)
        placed_raw = torch.cat([placed_features, placed_flag], dim=-1) 
        placed_emb = self.geom_proj(placed_raw) # [B, N_placed, d_model]

        # Máscara binaria para identificar qué bloques colocados son padding real
        placed_padding_mask = torch.all(placed_features == -1.0, dim=-1) # [B, N_placed]

        # Concatenamos el token estructural [CLS] en la posición 0
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, d_model]
        container_seq = torch.cat([cls_tokens, placed_emb], dim=1) # [B, 1 + N_placed, d_model]

        # Máscara de contenedor: [CLS] (índice 0) nunca es padding (False)
        cls_padding_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        container_padding_mask = torch.cat([cls_padding_mask, placed_padding_mask], dim=1) # [B, 1 + N_placed]

        # Procesamos la geometría construida (Auto-atención limpia)
        container_enriched = self.container_self_attn(container_seq, src_key_padding_mask=container_padding_mask)

        # --- 3. CROSS-ATTENTION BLINDADA CONTRA CONTAMINACIÓN DE PADDING ---
        # Creamos la máscara aditiva bidimensional [B, Q_len, K_len] -> [B, 1 + N_placed, N_avail]
        cross_attn_mask = container_padding_mask.unsqueeze(-1).expand(-1, -1, available_blocks.shape[1])
        
        # Máscara de flotantes: 0.0 significa atender, -inf significa ignorar matemáticamente
        cross_attn_mask_float = torch.zeros_like(cross_attn_mask, dtype=torch.float32)
        cross_attn_mask_float = cross_attn_mask_float.masked_fill(cross_attn_mask, float('-inf'))
        
        # 🚨 BLINDAJE INTERNO MATEMÁTICO DE FILAS DE ATTN_MASK:
        # Si una fila completa de la máscara es -inf (Query muerto), forzamos a que su primer elemento 
        # (columna 0) sea 0.0 de forma artificial. Esto salva al Softmax de la división por cero.
        # Al final, el output de esa fila basura será ignorado de todas formas por las capas posteriores, 
        # pero matemáticamente impide que se generen NaNs.
        cross_attn_mask_float[:, :, 0] = torch.where(
            container_padding_mask, # Si la posición del contenedor es padding
            torch.zeros_like(cross_attn_mask_float[:, :, 0]), # Ponemos un 0.0 de seguridad en la col 0
            cross_attn_mask_float[:, :, 0] # Si no es padding, dejamos su valor original
        )
        
        # Repetimos la máscara para cada una de las cabezas de atención (nhead) de forma consecutiva
        num_heads = self.container_self_attn.layers[0].self_attn.num_heads
        cross_attn_mask_float = cross_attn_mask_float.repeat_interleave(num_heads, dim=0)

        contextualized_scene, _ = self.inventory_cross_attn(
            query=container_enriched,
            key=inventory_embs,
            value=inventory_embs,
            key_padding_mask=inventory_padding_mask, # Máscara 1D para los Keys (Inventario)
            attn_mask=cross_attn_mask_float        # Máscara 2D corregida para los Queries
        ) # [B, 1 + N_placed, d_model]

        # --- 4. EXTRACCIÓN DEL TOKEN [CLS] Y REGRESIÓN DE VALOR ---
        # El índice 0 ahora concentra la abstracción pura de la escena combinatoria
        estado_representativo = contextualized_scene[:, 0, :] # [B, d_model]
        
        predicted_volume = self.value_mlp(estado_representativo).squeeze(-1) # [B]
        
        return predicted_volume
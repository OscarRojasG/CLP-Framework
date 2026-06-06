import numpy as np
from data.objects import *
from data.adapters.input.input_adapter import InputAdapter

class InputAdapterV10(InputAdapter):
    def __init__(self, max_blocks: int, max_pblocks: int, max_actions: int):
        # Actualizamos las dimensiones de los descriptores definitivos:
        # action_features: 10 descriptores
        # space_features: 13 descriptores (incluye density_ratio)
        # placed_features: 7 descriptores (6 de bounding box + 1 de contacto) 🚨
        super().__init__({
            "block_features": np.float32,
            "action_blocks": np.int32,
            "action_features": np.float32,
            "placed_features": np.float32,
            "space_features": np.float32
        }, max_blocks, max_pblocks)
        self.max_actions = max_actions

    def _compute_contact_flag(self, pb: PBlock, block: Block, space: Space) -> float:
        """
        Función auxiliar aislada para calcular si un bloque colocado 
        hace contacto físico directo con las caras del espacio residual.
        """
        epsilon = 1e-5
        
        touches_space = (
            # Contacto en Eje X (Izquierda o Derecha)
            abs((pb.x + block.l) - space.x) < epsilon or 
            abs(pb.x - (space.x + space.l)) < epsilon or
            
            # Contacto en Eje Y (Frente o Atrás)
            abs((pb.y + block.w) - space.y) < epsilon or 
            abs(pb.y - (space.y + space.w)) < epsilon or
            
            # Contacto en Eje Z (Abajo o Arriba)
            abs((pb.z + block.h) - space.z) < epsilon or 
            abs(pb.z - (space.z + space.h)) < epsilon
        )
        return 1.0 if touches_space else 0.0
    
    def enc_2_vec(self, blocks: list[Block]):
        block_features = np.full((self.max_blocks, 5), -1, dtype=np.float32)
        n_b = len(blocks)
        block_features[:n_b] = [[b.l, b.w, b.h, b.volume(), 1/b.n] for b in blocks[:n_b]]
        return (block_features, )
    
    def dec_2_vec(self, blocks: list[Block], space: Space, pblocks: list[PBlock], actions: list[Action]):
        action_blocks = np.full((self.max_actions,), -1, dtype=np.int32)
        action_features = np.full((self.max_actions, 10), -1, dtype=np.float32)
        
        # 🚨 CAMBIO ESTRUCTURAL: Inicializamos con 7 columnas por bloque colocado
        placed_features = np.full((self.max_pblocks, 7), -1, dtype=np.float32)
        
        # 1. Ingeniería de rasgos para el Espacio Residual (13 características)
        s_l, s_w, s_h = space.l, space.w, space.h
        s_vol = s_l * s_w * s_h
        ratio_lw = s_l / s_w if s_w > 0 else 1.0
        ratio_lh = s_l / s_h if s_h > 0 else 1.0
        total_placed_vol = sum(blocks[pb.id].volume() for pb in pblocks)
        density_ratio = total_placed_vol / (total_placed_vol + s_vol) if (total_placed_vol + s_vol) > 0 else 0.0

        space_features = np.array([
            space.x, space.y, space.z,
            space.x + s_l, space.y + s_w, space.z + s_h,
            s_l, s_w, s_h, s_vol, ratio_lw, ratio_lh, density_ratio
        ], dtype=np.float32)

        # 2. Ingeniería de rasgos para Acciones Candidatas (10 características)
        n_a = len(actions)
        action_blocks[:n_a] = [a.block_id for a in actions]
        for i, a in enumerate(actions):
            b = blocks[a.block_id]
            slack_l, slack_w, slack_h = s_l - b.l, s_w - b.w, s_h - b.h
            vol_occupancy = b.volume() / s_vol if s_vol > 0 else 0.0
            fit_l = b.l / s_l if s_l > 0 else 0.0
            fit_w = b.w / s_w if s_w > 0 else 0.0
            fit_h = b.h / s_h if s_h > 0 else 0.0
            
            action_features[i] = [
                a.loss if a.loss > 0 else 0.0, a.cs, a.norm_vcs(),
                slack_l, slack_w, slack_h, vol_occupancy, fit_l, fit_w, fit_h
            ]

        # 3. Procesamiento de Bloques Colocados con la función auxiliar (7 características) 🚨
        n_pb = len(pblocks)
        if n_pb > 0:
            for i, pb in enumerate(pblocks[:n_pb]):
                block = blocks[pb.id]
                
                # Invocamos la función auxiliar limpia
                contact_flag = self._compute_contact_flag(pb, block, space)
                
                # Almacenamos las 6 coordenadas originales + la nueva columna de contacto en la posición 6
                placed_features[i] = [
                    pb.x, pb.y, pb.z, 
                    pb.x + block.l, pb.y + block.w, pb.z + block.h,
                    contact_flag
                ]
            
        return (
            action_blocks,
            action_features,
            placed_features,
            space_features
        )
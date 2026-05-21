import numpy as np
from data.objects import *
from data.adapters.input.input_adapter import InputAdapter

class InputAdapterV6(InputAdapter):
    def __init__(self, max_blocks: int, max_pblocks: int, max_actions: int):
        super().__init__({
            "block_features": np.float32,
            "action_blocks": np.int32,
            "action_features": np.float32,
            "placed_features": np.float32,
            "space_features": np.float32,
            "available_blocks": np.int32
        }, max_blocks, max_pblocks)
        self.max_actions = max_actions
    
    def enc_2_vec(self, blocks: list[Block]):
        block_features = np.full((self.max_blocks, 5), -1, dtype=np.float32)

        n_b = len(blocks)
        block_features[:n_b] = [[b.l, b.w, b.h, b.volume(), 1/b.n] for b in blocks[:n_b]]

        return (block_features, )
    
    def dec_2_vec(self, blocks: list[Block], space: Space, pblocks: list[PBlock], actions: list[Action]):
        action_blocks = np.full((self.max_actions,), -1, dtype=np.int32)
        action_features = np.full((self.max_actions, 2), -1, dtype=np.float32)
        placed_features = np.full((self.max_pblocks, 6), -1, dtype=np.float32)
        space_features = np.array([space.x, space.y, space.z, space.x + space.l, space.y + space.w, space.z + space.h], dtype=np.float32)

        # 1. Procesamiento de acciones candidato del paso actual
        n_a = len(actions)
        action_blocks[:n_a] = [a.block_id for a in actions]
        action_features[:n_a] = [[a.loss if a.loss > 0 else 0, a.cs] for a in actions]

        # 2. Procesamiento de bloques ya colocados
        n_pb = len(pblocks)
        if n_pb > 0:
            for i, pb in enumerate(pblocks[:n_pb]):
                block = blocks[pb.id]
                placed_features[i] = [pb.x, pb.y, pb.z, pb.x + block.l, pb.y + block.w, pb.z + block.h] 
            
        # 3. CÁLCULO DE BLOQUES DISPONIBLES GEOMÉTRICAMENTE (Con blindaje Epsilon)
        available_blocks = np.full((self.max_blocks,), -1, dtype=np.int32)
        
        # Tolerancia para absorber imprecisiones acumuladas de punto flotante
        eps = 1e-5
        
        valid_ids = []
        for b_id, b in enumerate(blocks):
            # Sumamos epsilon a las cotas del espacio para asegurar que variaciones
            # microscópicas negativas no dejen fuera a bloques legítimos
            if b.l <= (space.l + eps) and b.w <= (space.w + eps) and b.h <= (space.h + eps):
                valid_ids.append(b_id)
                
        n_av = len(valid_ids)
        if n_av > 0:
            limit = min(n_av, self.max_blocks)
            available_blocks[:limit] = valid_ids[:limit]

        return (
            action_blocks,
            action_features,
            placed_features,
            space_features,
            available_blocks
        )
import numpy as np
from data.objects import *
from data.adapters.input.input_adapter import InputAdapter

class InputAdapterV11(InputAdapter):
    def __init__(self, max_blocks: int, max_pblocks: int, max_actions: int):
        # Agregamos la clave placed_blocks para cumplir con el esquema de tensores
        super().__init__({
            "block_features": np.float32,
            "action_blocks": np.int32,
            "action_features": np.float32,
            "space_features": np.float32,
            "placed_blocks": np.int32,      # 🚨 NUEVO: Tipo int32 para IDs
            "placed_features": np.float32,
            "vcs": np.float32
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
        space_features = np.array([space.x, space.y, space.z, space.x + space.l, space.y + space.w, space.z + space.h], dtype=np.float32)
        
        # 🚨 NUEVOS: Inicializadores para componentes de bloques colocados
        placed_blocks = np.full((self.max_pblocks,), -1, dtype=np.int32)
        placed_features = np.full((self.max_pblocks, 6), -1, dtype=np.float32)

        n_a = len(actions)
        action_blocks[:n_a] = [a.block_id for a in actions]
        action_features[:n_a] = [[a.loss if a.loss > 0 else 0, a.cs] for a in actions]

        n_pb = len(pblocks)
        if n_pb > 0:
            placed_blocks[:n_pb] = [pb.id for pb in pblocks[:n_pb]] # Guardamos IDs originales
            for i, pb in enumerate(pblocks[:n_pb]):
                block = blocks[pb.id]
                placed_features[i] = [pb.x, pb.y, pb.z, pb.x + block.l, pb.y + block.w, pb.z + block.h]

        vcs = np.full((self.max_actions,), -1, dtype=np.float32)
        n_a = len(actions)
        vcs[:n_a] = [a.calc_vcs() for a in actions]
            
        # El retorno debe acoplarse con el orden estricto esperado en el forward
        return (
            action_blocks,
            action_features,
            space_features,
            placed_blocks,
            placed_features,
            vcs
        )
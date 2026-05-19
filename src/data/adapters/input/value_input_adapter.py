import numpy as np
from data.objects import *
from data.adapters.input.input_adapter import InputAdapter

class ValueInputAdapter(InputAdapter):
    def __init__(self, max_blocks: int, max_pblocks: int):
        super().__init__({
            "block_features": np.float32,
            "placed_blocks": np.int32,
            "placed_features": np.float32,
            "space_features": np.float32,
        }, max_blocks, max_pblocks)
    
    def enc_2_vec(self, blocks: list[Block]):
        block_features = np.full((self.max_blocks, 8), -1, dtype=np.float32)

        n_b = len(blocks)
        block_features[:n_b] = [[b.l, b.w, b.h, b.volume(), 1/b.n, b.lw(), b.lh(), b.wh()] for b in blocks[:n_b]]

        return (block_features, )
    
    def dec_2_vec(self, blocks: list[Block], space: Space, pblocks: list[PBlock], actions: list[Action]):
        placed_blocks = np.full((self.max_pblocks,), -1, dtype=np.int32)
        placed_features = np.full((self.max_pblocks, 3), -1, dtype=np.float32)
        space_features = np.array([space.x, space.y, space.z, space.l, space.w, space.h], dtype=np.float32)

        n_pb = len(pblocks)
        if n_pb > 0:
            placed_blocks[:n_pb] = [pb.id for pb in pblocks[:n_pb]]
            placed_features[:n_pb] = [[pb.x, pb.y, pb.z] for pb in pblocks[:n_pb]]
            
        return (
            placed_blocks,
            placed_features,
            space_features
        )
import numpy as np
from data.objects import *
from data.adapters.input.input_adapter import InputAdapter

class InputAdapterV3(InputAdapter):
    def __init__(self, max_blocks: int, max_pblocks: int, max_actions: int):
        super().__init__({
            "block_features": np.float32,
            "action_blocks": np.int32,
            "placed_features": np.float32,
            "space_features": np.float32
        }, max_blocks, max_pblocks)
        self.max_actions = max_actions
    
    def enc_2_vec(self, blocks: list[Block]):
        block_features = np.full((self.max_blocks, 5), -1, dtype=np.float32)

        n_b = len(blocks)
        block_features[:n_b] = [[b.l, b.w, b.h, b.volume(), 1/b.n] for b in blocks[:n_b]]

        return (block_features, )
    
    def dec_2_vec(self, blocks: list[Block], space: Space, pblocks: list[PBlock], actions: list[Action]):
        action_blocks = np.full((self.max_actions,), -1, dtype=np.int32)
        placed_features = np.full((self.max_pblocks, 6), -1, dtype=np.float32)
        space_features = np.array([space.x, space.y, space.z, space.x + space.l, space.y + space.w, space.z + space.h], dtype=np.float32)

        n_a = len(actions)
        action_blocks[:n_a] = [a.block_id for a in actions]

        n_pb = len(pblocks)
        if n_pb > 0:
            for pb in pblocks[:n_pb]:
                block = blocks[pb.id]
                placed_features[:n_pb] = [pb.x, pb.y, pb.z, pb.x + block.l, pb.y + block.w, pb.z + block.h] 
            
        return (
            action_blocks,
            placed_features,
            space_features
        )
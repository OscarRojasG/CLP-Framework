import numpy as np
from data.objects import *
from data.adapters.input.input_adapter import InputAdapter

class InputAdapterV1(InputAdapter):
    def __init__(self, max_blocks: int, max_pblocks: int, max_actions: int):
        # Actualizamos el diccionario del constructor con las nuevas llaves semánticas
        super().__init__({
            "block_features": np.float32,
            "action_blocks": np.int32,
            "action_coords": np.float32,
            "placed_coords": np.float32,
            "space_coords": np.float32,
            "action_features": np.float32
        }, max_blocks, max_pblocks)
        self.max_actions = max_actions
    
    def enc_2_vec(self, blocks: list[Block]):
        block_features = np.full((self.max_blocks, 5), -1, dtype=np.float32)

        n_b = len(blocks)
        block_features[:n_b] = [[b.l, b.w, b.h, b.volume(), 1/b.n] for b in blocks[:n_b]]

        return (block_features, )
    
    def dec_2_vec(self, blocks: list[Block], space: Space, pblocks: list[PBlock], actions: list[Action]):
        action_blocks = np.full((self.max_actions,), -1, dtype=np.int32)
        action_coords = np.full((self.max_actions, 6), -1, dtype=np.float32)
        placed_coords = np.full((self.max_pblocks, 6), -1, dtype=np.float32)
        action_features = np.full((self.max_actions, 2), -1, dtype=np.float32)

        # Espacio — normalizado (consistente con el resto)
        space_coords = np.array([
            space.x, space.y, space.z,
            space.x + space.l, space.y + space.w, space.z + space.h
        ], dtype=np.float32)

        n_a = len(actions)
        action_blocks[:n_a] = [a.block_id for a in actions]
        for i, a in enumerate(actions[:n_a]):
            block = blocks[a.block_id]
            action_coords[i] = [
                space.x, space.y, space.z,
                space.x + block.l, space.y + block.w, space.z + block.h
            ]
            action_features[i] = [a.loss if a.loss > 0 else 0, a.cs]

        n_pb = len(pblocks)
        if n_pb > 0:
            for i, pb in enumerate(pblocks[:n_pb]):
                block = blocks[pb.id]
                placed_coords[i] = [
                    pb.x, pb.y, pb.z,
                    pb.x + block.l, pb.y + block.w, pb.z + block.h
                ]

        return (
            action_blocks,
            action_coords,
            placed_coords,
            space_coords,
            action_features
        )
import numpy as np
from data.objects import *
from data.adapters.input.input_adapter import InputAdapter

class InputAdapterV2(InputAdapter):
    def __init__(self, max_boxes: int, max_blocks: int, max_pblocks: int, max_actions: int):
        # Actualizamos el diccionario del constructor con las nuevas llaves semánticas
        super().__init__({
            "block_features": np.float32,
            "box_features": np.float32,
            "total_boxes": np.int32,
            "boxes_per_block": np.int32,
            "action_blocks": np.int32,
            "action_features": np.float32,
            "placed_blocks": np.int32,
            "placed_features": np.float32,
        }, max_blocks, max_pblocks)
        self.max_actions = max_actions
        self.max_boxes = max_boxes
    
    def enc_2_vec(self, boxes: list[Box], blocks: list[Block]):
        block_features = np.full((self.max_blocks, 5), -1, dtype=np.float32)
        box_features = np.full((self.max_boxes, 3), -1, dtype=np.float32)
        total_boxes = np.full((self.max_boxes,), -1, dtype=np.int32)
        boxes_per_block = np.zeros((self.max_blocks, self.max_boxes), dtype=np.int32)

        # Llenado de features
        n_b = len(blocks)
        block_features[:n_b] = [[b.l, b.w, b.h, b.volume(), 1/b.n] for b in blocks[:n_b]]

        n_x = len(boxes)
        box_features[:n_x] = [[b.l, b.w, b.h] for b in boxes[:n_x]]        
        total_boxes[:n_x] = [b.n for b in boxes[:n_x]] # <--- Llenado de total_boxes

        for i, block in enumerate(blocks[:self.max_blocks]):
            for box_id, quantity in block.boxes.items():
                boxes_per_block[i, box_id] = quantity 

        return (block_features, box_features, total_boxes, boxes_per_block)
    
    def dec_2_vec(self, boxes: list[Box], blocks: list[Block], space: Space, pblocks: list[PBlock], actions: list[Action]):
        action_blocks = np.full((self.max_actions,), -1, dtype=np.int32)
        action_features = np.full((self.max_actions, 2), -1, dtype=np.float32)
        
        placed_blocks = np.full((self.max_pblocks,), -1, dtype=np.int32)
        placed_features = np.full((self.max_pblocks, 6), -1, dtype=np.float32)

        # 1. Acciones Candidatas (Identidad + Desempeño)
        n_a = len(actions)
        action_blocks[:n_a] = [a.block_id for a in actions]
        for i, a in enumerate(actions[:n_a]):
            action_features[i] = [a.loss if a.loss > 0 else 0, a.cs]

        # 2. Entorno Relativo de Bloques Colocados
        n_pb = len(pblocks)
        placed_blocks[:n_pb] = [pb.id for pb in pblocks[:n_pb]]
        if n_pb > 0:
            for i, pb in enumerate(pblocks[:n_pb]):
                block = blocks[pb.id]
                
                # Todo se mide como un desplazamiento respecto al inicio de la zona de juego actual
                rel_x1 = pb.x - space.x
                rel_y1 = pb.y - space.y
                rel_z1 = pb.z - space.z
                
                rel_x2 = (pb.x + block.l) - space.x
                rel_y2 = (pb.y + block.w) - space.y
                rel_z2 = (pb.z + block.h) - space.z
                
                placed_features[i] = [rel_x1, rel_y1, rel_z1, rel_x2, rel_y2, rel_z2]

        return (
            action_blocks,
            action_features,
            placed_blocks,
            placed_features,
        )
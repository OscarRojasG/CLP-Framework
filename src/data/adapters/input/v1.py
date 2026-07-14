import numpy as np
from data.objects import *
from data.adapters.input.input_adapter import InputAdapter

class InputAdapterV1(InputAdapter):
    def __init__(self, max_blocks: int, max_pblocks: int, max_actions: int):
        super().__init__({
            "block_features": np.float32,
            "action_blocks": np.int32,
            "action_features": np.float32,
            "placed_features": np.float32
        }, max_blocks, max_pblocks)
        self.max_actions = max_actions
    
    """
    Transformación vectorial para la arquitectura de codificación.
    
    Retorna: (block_features,)
    Features: Representación estática de los bloques disponibles. Incorpora las 
    dimensiones espaciales (l, w, h), el volumen total y la inversa de la cantidad
    de cajas que lo componen (1/n).
    """
    def enc_2_vec(self, blocks: list[Block]):
        block_features = np.full((self.max_blocks, 5), -1, dtype=np.float32)

        n_b = len(blocks)
        block_features[:n_b] = [[b.l, b.w, b.h, b.volume(), 1/b.n] for b in blocks[:n_b]]

        return (block_features, )
    
    """
    Transformación vectorial para la arquitectura de decodificación.
    
    Retorna: (action_blocks, action_features, placed_features)
    Features: 
    * action_blocks: Identificadores (IDs) de los bloques candidatos a colocar.
    * action_features: Métricas de desempeño de las acciones (loss umbralizada a 0 y cs).
    * placed_features: Entorno geométrico relativo (x1, y1, z1, x2, y2, z2) de los 
      bloques ya empaquetados, calculados como desplazamiento espacial puro respecto 
      a los límites de la zona de carga (Space) actual.
    """
    def dec_2_vec(self, blocks: list[Block], space: Space, pblocks: list[PBlock], actions: list[Action]):
        action_blocks = np.full((self.max_actions,), -1, dtype=np.int32)
        action_features = np.full((self.max_actions, 2), -1, dtype=np.float32)
        placed_features = np.full((self.max_pblocks, 6), -1, dtype=np.float32)

        n_pb = len(pblocks)
        if n_pb > 0:
            for i, pb in enumerate(pblocks[:n_pb]):
                block = blocks[pb.id]
                
                rel_x1 = pb.x - space.x
                rel_y1 = pb.y - space.y
                rel_z1 = pb.z - space.z
                
                rel_x2 = (pb.x + block.l) - space.x
                rel_y2 = (pb.y + block.w) - space.y
                rel_z2 = (pb.z + block.h) - space.z
                
                placed_features[i] = [rel_x1, rel_y1, rel_z1, rel_x2, rel_y2, rel_z2]

        n_a = len(actions)
        action_blocks[:n_a] = [a.block_id for a in actions]
        for i, a in enumerate(actions[:n_a]):
            action_features[i] = [a.loss if a.loss > 0 else 0, a.cs]

        return (
            action_blocks,
            action_features,
            placed_features
        )
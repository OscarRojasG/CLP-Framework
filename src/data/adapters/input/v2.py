import numpy as np
from data.objects import *
from data.adapters.input.input_adapter import InputAdapter

class InputAdapterV2(InputAdapter):
    def __init__(self, max_blocks: int, max_pblocks: int, max_actions: int):
        super().__init__({
            "block_features": np.float32,
            "action_blocks": np.int32,
            "action_features": np.float32,
            "placed_blocks": np.int32,
            "placed_features": np.float32,
            "space_features": np.float32,
        }, max_blocks, max_pblocks, max_actions)

    def input_2_vec(self, blocks: list[Block], space: Space, pblocks: list[PBlock], actions: list[Action]):
        enc_data = self.enc_2_vec(blocks)
        dec_data = self.dec_2_vec(space, pblocks, actions)

        return (*enc_data, *dec_data)
    
    def enc_2_vec(self, blocks: list[Block]):
        block_features = np.full((self.max_blocks, 8), -1, dtype=np.float32)

        n_b = len(blocks)
        block_features[:n_b] = [[b.l, b.w, b.h, b.volume(), 1/b.n, b.lw(), b.wh(), b.lh()] for b in blocks[:n_b]]

        return (block_features, )
    
    def dec_2_vec(self, blocks: list[Block], space: Space, pblocks: list[PBlock], actions: list[Action]):
        action_blocks = np.full((self.max_actions,), -1, dtype=np.int32)
        action_features = np.full((self.max_actions, 2), -1, dtype=np.float32)
        placed_blocks = np.full((self.max_pblocks,), -1, dtype=np.int32)
        placed_features = np.full((self.max_pblocks, 4), -1, dtype=np.float32)
        space_features = np.array([space.x, space.y, space.z, space.l, space.w, space.h], dtype=np.float32)

        n_a = len(actions)
        action_blocks[:n_a] = [a.block_id for a in actions]
        action_features[:n_a] = [[a.loss if a.loss > 0 else 0, a.cs] for a in actions]

        n_pb = len(pblocks)
        if n_pb > 0:
            placed_blocks[:n_pb] = [pb.id for pb in pblocks]

            features_list = []
            for pb in pblocks:
                # Accedemos a las dimensiones del bloque usando su ID
                b_dim = blocks[pb.id]
                
                # Verificamos si el bloque está pegado a las caras externas de este 'space'
                touches_space = (
                    # Contacto en Eje X (Izquierda o Derecha)
                    (pb.x + b_dim.l) == space.x or pb.x == (space.x + space.l) or
                    # Contacto en Eje Y (Frente o Atrás)
                    (pb.y + b_dim.w) == space.y or pb.y == (space.y + space.w) or
                    # Contacto en Eje Z (Abajo o Arriba)
                    (pb.z + b_dim.h) == space.z or pb.z == (space.z + space.h)
                )
                
                # Convertimos el booleano a 1.0 o 0.0
                contact_flag = 1.0 if touches_space else 0.0
                
                # Guardamos las 4 features
                features_list.append([pb.x, pb.y, pb.z, contact_flag])
            
            # Asignamos la matriz de características generada
            placed_features[:n_pb] = features_list
            
        return (
            action_blocks,
            action_features,
            placed_blocks,
            placed_features,
            space_features
        )
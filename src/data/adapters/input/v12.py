import numpy as np
from data.objects import *
from data.adapters.input.input_adapter import InputAdapter
import math

class InputAdapterV12(InputAdapter):
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
        action_features = np.full((self.max_actions, 7), -1, dtype=np.float32)  # loss + 6 cs direccionales
        space_features = np.array([space.x, space.y, space.z, space.x + space.l, space.y + space.w, space.z + space.h], dtype=np.float32)
        
        placed_blocks = np.full((self.max_pblocks,), -1, dtype=np.int32)
        placed_features = np.full((self.max_pblocks, 12), -1, dtype=np.float32)  # 6 coords + 6 cs direccionales + 1 cs agregado de C++... no, solo 12: coords + cs dir

        L_CONT = 587.0
        W_CONT = 233.0
        H_CONT = 220.0
        p = 0.04

        def cs_por_cara(bx1, by1, bz1, bx2, by2, bz2, placed_list):
            p = 0.04
            cont_L, cont_W, cont_H = 587.0, 233.0, 220.0

            b_L = bx2 - bx1; b_W = by2 - by1; b_H = bz2 - bz1
            bb_surface = 2.0 * (b_W * b_H + b_L * b_H + b_L * b_W)
            if bb_surface == 0:
                return [0.0] * 6

            bexp = {
                'Xmin': bx1 - math.ceil(p * b_L), 'Xmax': bx2 + math.ceil(p * b_L),
                'Ymin': by1 - math.ceil(p * b_W), 'Ymax': by2 + math.ceil(p * b_W),
                'Zmin': bz1 - math.ceil(p * b_H), 'Zmax': bz2 + math.ceil(p * b_H),
            }

            # [x_izq, x_der, y_frente, y_atras, z_abajo, z_arriba]
            faces = [0.0] * 6

            for pb in placed_list:
                if pb[0] == -1:
                    continue
                px1, py1, pz1, px2, py2, pz2 = pb

                if (bexp['Xmax'] < px1 or px2 < bexp['Xmin'] or
                    bexp['Ymax'] < py1 or py2 < bexp['Ymin'] or
                    bexp['Zmax'] < pz1 or pz2 < bexp['Zmin']):
                    continue

                ox1 = max(bx1, px1); ox2 = min(bx2, px2)
                oy1 = max(by1, py1); oy2 = min(by2, py2)
                oz1 = max(bz1, pz1); oz2 = min(bz2, pz2)

                ov_x = ox2 - ox1; ov_y = oy2 - oy1; ov_z = oz2 - oz1

                s = 0.0
                face_idx = -1

                # Misma lógica mutuamente excluyente del original
                if s == 0.0 and ov_y > 0 and ov_z > 0:
                    if bx2 >= px1 - p * b_L:
                        s = ov_y * ov_z; face_idx = 1  # x_der
                    elif px2 >= bx1 - p * b_L:
                        s = ov_y * ov_z; face_idx = 0  # x_izq

                if s == 0.0 and ov_x > 0 and ov_z > 0:
                    if by2 >= py1 - p * b_W:
                        s = ov_x * ov_z; face_idx = 3  # y_atras
                    elif py2 >= by1 - p * b_W:
                        s = ov_x * ov_z; face_idx = 2  # y_frente

                if s == 0.0 and ov_x > 0 and ov_y > 0:
                    if bz2 >= pz1 - p * b_H:
                        s = ov_x * ov_y; face_idx = 5  # z_arriba
                    elif pz2 >= bz1 - p * b_H:
                        s = ov_x * ov_y; face_idx = 4  # z_abajo

                if s == 0.0:
                    if ox2 == ox1 and ov_y > 0 and ov_z > 0:
                        if bx2 >= px1 - p * b_L or px2 >= bx1 - p * b_L:
                            s = ov_y * ov_z; face_idx = 0
                    elif oy2 == oy1 and ov_x > 0 and ov_z > 0:
                        if by2 >= py1 - p * b_W or py2 >= by1 - p * b_W:
                            s = ov_x * ov_z; face_idx = 2
                    elif oz2 == oz1 and ov_x > 0 and ov_y > 0:
                        if bz2 >= pz1 - p * b_H or pz2 >= bz1 - p * b_H:
                            s = ov_x * ov_y; face_idx = 4

                if face_idx >= 0:
                    faces[face_idx] += s

            # Paredes del contenedor
            if bx1 <= p * b_L:                  faces[0] += b_W * b_H
            if bx2 >= cont_L - p * b_L:         faces[1] += b_W * b_H
            if by1 <= p * b_W:                  faces[2] += b_L * b_H
            if by2 >= cont_W - p * b_W:         faces[3] += b_L * b_H
            if bz1 <= p * b_H:                  faces[4] += b_L * b_W
            if bz2 >= cont_H - p * b_H:         faces[5] += b_L * b_W

            # Normalizar cada cara por su área propia
            face_areas = [b_W*b_H, b_W*b_H, b_L*b_H, b_L*b_H, b_L*b_W, b_L*b_W]
            return [f / a if a > 0 else 0.0 for f, a in zip(faces, face_areas)]

        # --- Placed blocks en desnormalizado para reusar en acciones ---
        placed_coords_raw = []
        n_pb = len(pblocks)
        if n_pb > 0:
            placed_blocks[:n_pb] = [pb.id for pb in pblocks[:n_pb]]
            for i, pb in enumerate(pblocks[:n_pb]):
                block = blocks[pb.id]
                # Desnormalizar
                px1 = round(pb.x * L_CONT); py1 = round(pb.y * L_CONT); pz1 = round(pb.z * L_CONT)
                px2 = round((pb.x + block.l) * L_CONT)
                py2 = round((pb.y + block.w) * L_CONT)
                pz2 = round((pb.z + block.h) * L_CONT)
                placed_coords_raw.append([px1, py1, pz1, px2, py2, pz2])

                # CS direccional del bloque colocado respecto al espacio actual
                sx1 = round(space.x * L_CONT); sy1 = round(space.y * L_CONT); sz1 = round(space.z * L_CONT)
                sx2 = round((space.x + space.l) * L_CONT)
                sy2 = round((space.y + space.w) * L_CONT)
                sz2 = round((space.z + space.h) * L_CONT)

                cs_dir = cs_por_cara(px1, py1, pz1, px2, py2, pz2, [[sx1, sy1, sz1, sx2, sy2, sz2]])
                placed_features[i] = [pb.x, pb.y, pb.z, pb.x + block.l, pb.y + block.w, pb.z + block.h] + cs_dir

        # Desnormalizar espacio (igual para todas las acciones)
        sx1 = round(space.x * L_CONT); sy1 = round(space.y * L_CONT); sz1 = round(space.z * L_CONT)

        n_a = len(actions)
        action_blocks[:n_a] = [a.block_id for a in actions]
        for i, a in enumerate(actions[:n_a]):
            block = blocks[a.block_id]
            ax1 = sx1; ay1 = sy1; az1 = sz1
            ax2 = sx1 + round(block.l * L_CONT)
            ay2 = sy1 + round(block.w * L_CONT)
            az2 = sz1 + round(block.h * L_CONT)
            
            cs_dir = cs_por_cara(ax1, ay1, az1, ax2, ay2, az2, placed_coords_raw)
            loss = a.loss if a.loss > 0 else 0.0
            action_features[i] = [loss] + cs_dir

        vcs = np.full((self.max_actions,), -1, dtype=np.float32)
        vcs[:n_a] = [a.calc_vcs() for a in actions[:n_a]]

        return (
            action_blocks,
            action_features,
            space_features,
            placed_blocks,
            placed_features,
            vcs
        )
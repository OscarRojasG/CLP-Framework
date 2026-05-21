import numpy as np
from data.objects import *
from data.adapters.input.input_adapter import InputAdapter

class InputAdapterV9(InputAdapter):
    def __init__(self, max_blocks: int, max_pblocks: int, max_actions: int):
        # Actualizamos el diccionario del constructor con las nuevas llaves semánticas
        super().__init__({
            "block_features": np.float32,
            "action_blocks": np.int32,
            "action_coords": np.float32,   # Renombrado de action_features (6 coords)
            "placed_coords": np.float32,   # Renombrado de placed_features (6 coords)
            "action_features": np.float32   # Reservado para la heurística de emergencia (loss, cs)
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

        n_a = len(actions)
        action_blocks[:n_a] = [a.block_id for a in actions]
        
        for i, a in enumerate(actions[:n_a]):
            block = blocks[a.block_id]
            action_coords[i] = [
                space.x, 
                space.y, 
                space.z, 
                space.x + block.l, 
                space.y + block.w, 
                space.z + block.h
            ]
            action_coords[i] = [round(x * 587) for x in action_coords[i]]
            #action_features[i] = [a.loss if a.loss > 0 else 0, a.cs]

        n_pb = len(pblocks)
        if n_pb > 0:
            for i, pb in enumerate(pblocks[:n_pb]):
                block = blocks[pb.id]
                placed_coords[i] = [pb.x, pb.y, pb.z, pb.x + block.l, pb.y + block.w, pb.z + block.h]
                #placed_coords[i] = [round(x * 587) for x in placed_coords[i]]

        #debug_cs(action_coords[2], placed_coords, actions[2].cs)
                     
        return (
            action_blocks,
            action_coords,
            placed_coords,
            action_features
        )

def debug_cs(block_coords, placed_coords, target_cs):
    p = 0.04
    cont_L, cont_W, cont_H = 587.0, 233.0, 220.0
    
    b = {
        'Xmin': float(block_coords[0]), 'Ymin': float(block_coords[1]), 'Zmin': float(block_coords[2]),
        'Xmax': float(block_coords[3]), 'Ymax': float(block_coords[4]), 'Zmax': float(block_coords[5])
    }
    b_L = b['Xmax'] - b['Xmin']
    b_W = b['Ymax'] - b['Ymin']
    b_H = b['Zmax'] - b['Zmin']

    # bexp float, sin floor
    import math
    bexp = {
        'Xmin': b['Xmin'] - math.ceil(p * b_L),
        'Ymin': b['Ymin'] - math.ceil(p * b_W),
        'Zmin': b['Zmin'] - math.ceil(p * b_H),
        'Xmax': b['Xmax'] + math.ceil(p * b_L),
        'Ymax': b['Ymax'] + math.ceil(p * b_W),
        'Zmax': b['Zmax'] + math.ceil(p * b_H),
    }

    surface = 0.0

    for idx, pb in enumerate(placed_coords):
        if pb[0] == -1:
            continue

        bi = {
            'Xmin': float(pb[0]), 'Ymin': float(pb[1]), 'Zmin': float(pb[2]),
            'Xmax': float(pb[3]), 'Ymax': float(pb[4]), 'Zmax': float(pb[5])
        }

        # Filtro bexp float con < (criterio Bullet: tocar = intersectar)
        if (bexp['Xmax'] < bi['Xmin'] or bi['Xmax'] < bexp['Xmin'] or
            bexp['Ymax'] < bi['Ymin'] or bi['Ymax'] < bexp['Ymin'] or
            bexp['Zmax'] < bi['Zmin'] or bi['Zmax'] < bexp['Zmin']):
            continue

        x_min = max(b['Xmin'], bi['Xmin']); x_max = min(b['Xmax'], bi['Xmax'])
        y_min = max(b['Ymin'], bi['Ymin']); y_max = min(b['Ymax'], bi['Ymax'])
        z_min = max(b['Zmin'], bi['Zmin']); z_max = min(b['Zmax'], bi['Zmax'])

        s = 0.0

        # Guardas estrictas con >, caso exacto con ==
        if s == 0.0 and (y_max > y_min) and (z_max > z_min):
            if b['Xmax'] >= bi['Xmin'] - p * b_L:
                s = (y_max - y_min) * (z_max - z_min)
            elif bi['Xmax'] >= b['Xmin'] - p * b_L:
                s = (y_max - y_min) * (z_max - z_min)

        if s == 0.0 and (x_max > x_min) and (z_max > z_min):
            if b['Ymax'] >= bi['Ymin'] - p * b_W:
                s = (x_max - x_min) * (z_max - z_min)
            elif bi['Ymax'] >= b['Ymin'] - p * b_W:
                s = (x_max - x_min) * (z_max - z_min)

        if s == 0.0 and (x_max > x_min) and (y_max > y_min):
            if b['Zmax'] >= bi['Zmin'] - p * b_H:
                s = (x_max - x_min) * (y_max - y_min)
            elif bi['Zmax'] >= b['Zmin'] - p * b_H:
                s = (x_max - x_min) * (y_max - y_min)

        # Contacto exacto en un eje (overlap == 0 en ese eje)
        if s == 0.0:
            if x_max == x_min and y_max > y_min and z_max > z_min:
                if b['Xmax'] >= bi['Xmin'] - p * b_L or bi['Xmax'] >= b['Xmin'] - p * b_L:
                    s = (y_max - y_min) * (z_max - z_min)
            elif y_max == y_min and x_max > x_min and z_max > z_min:
                if b['Ymax'] >= bi['Ymin'] - p * b_W or bi['Ymax'] >= b['Ymin'] - p * b_W:
                    s = (x_max - x_min) * (z_max - z_min)
            elif z_max == z_min and x_max > x_min and y_max > y_min:
                if b['Zmax'] >= bi['Zmin'] - p * b_H or bi['Zmax'] >= b['Zmin'] - p * b_H:
                    s = (x_max - x_min) * (y_max - y_min)

        surface += s

    # Paredes contenedor
    wsurface = 0.0
    if b['Xmin'] <= p * b_L:        wsurface += b_W * b_H
    if b['Xmax'] >= cont_L - p * b_L: wsurface += b_W * b_H
    if b['Ymin'] <= p * b_W:        wsurface += b_L * b_H
    if b['Ymax'] >= cont_W - p * b_W: wsurface += b_L * b_H
    if b['Zmin'] <= p * b_H:        wsurface += b_L * b_W
    if b['Zmax'] >= cont_H - p * b_H: wsurface += b_L * b_W
    surface += wsurface

    bb_surface = 2.0 * (b_W * b_H + b_L * b_H + b_L * b_W)
    cs_calculado = surface / bb_surface
    diff = abs(target_cs - cs_calculado)
    status = "✅ MATCH" if diff < 1e-4 else "❌ DISCREPANCIA"

    print(f"\n  walls_surface={wsurface}, total surface={surface}, bb_surface={bb_surface}")
    print(f"  Target={target_cs:.8f} | Python={cs_calculado:.8f} | {status} (diff={diff:.8f})\n")
    
'''
def debug_hcs(block_coords, placed_coords, target_cs):
    p = 0.04
    cont_L, cont_W, cont_H = 587.0, 233.0, 220.0
    
    # 1. Definición exacta de la caja bb (b)
    b = {
        'Xmin': float(block_coords[0]),
        'Ymin': float(block_coords[1]),
        'Zmin': float(block_coords[2]),
        'Xmax': float(block_coords[3]),
        'Ymax': float(block_coords[4]),
        'Zmax': float(block_coords[5])
    }
    
    b_L = b['Xmax'] - b['Xmin']
    b_W = b['Ymax'] - b['Ymin']
    b_H = b['Zmax'] - b['Zmin']
    
    # 2. Construcción matemática de la caja expandida bexp (oo - diff, oo + b + diff)
    diff_x = p * b_L
    diff_y = p * b_W
    diff_z = p * b_H
    
    bexp = {
        'Xmin': b['Xmin'] - diff_x,
        'Ymin': b['Ymin'] - diff_y,
        'Zmin': b['Zmin'] - diff_z,
        'Xmax': b['Xmax'] + diff_x,
        'Ymax': b['Ymax'] + diff_y,
        'Zmax': b['Zmax'] + diff_z
    }
    
    surface = 0.0

    # 3. Bucle analítico sobre bloques colocados (get_intersected_objects)
    for pb in placed_coords:
        if pb[0] == -1 or (pb[0] == 0 and pb[3] == 0):
            continue
            
        bi = {
            'Xmin': float(pb[0]),
            'Ymin': float(pb[1]),
            'Zmin': float(pb[2]),
            'Xmax': float(pb[3]),
            'Ymax': float(pb[4]),
            'Zmax': float(pb[5])
        }
        
        # Filtro estricto AABB de Bullet contra la caja expandida bexp
        if (bexp['Xmax'] <= bi['Xmin'] or bi['Xmax'] <= bexp['Xmin'] or
            bexp['Ymax'] <= bi['Ymin'] or bi['Ymax'] <= bexp['Ymin'] or
            bexp['Zmax'] <= bi['Zmin'] or bi['Zmax'] <= bexp['Zmin']):
            continue
            
        x_min = max(b['Xmin'], bi['Xmin'])
        x_max = min(b['Xmax'], bi['Xmax'])
        y_min = max(b['Ymin'], bi['Ymin'])
        y_max = min(b['Ymax'], bi['Ymax'])
        z_min = max(b['Zmin'], bi['Zmin'])
        z_max = min(b['Zmax'], bi['Zmax'])
        
        s = 0.0
        
        # ✅ CAMBIO CLAVE: Evaluamos cada eje de forma independiente usando 'if' separados
        # Esto evita que una condición falsa dentro del bloque X aborte la revisión en Y o Z.
        # Una vez que un eje registra contacto real (s > 0), rompemos para pasar al siguiente bloque.
        
        # --- EJE X ---
        if (y_max > y_min) and (z_max > z_min) and (s == 0.0):
            if b['Xmax'] >= bi['Xmin'] - p * b_L:
                s = (y_max - y_min) * (z_max - z_min)
            elif bi['Xmax'] >= b['Xmin'] - p * b_L:
                s = (y_max - y_min) * (z_max - z_min)
                
        # --- EJE Y ---
        if (x_max > x_min) and (z_max > z_min) and (s == 0.0):
            if b['Ymax'] >= bi['Ymin'] - p * b_W:
                s = (x_max - x_min) * (z_max - z_min)
            elif bi['Ymax'] >= b['Ymin'] - p * b_W:
                s = (x_max - x_min) * (z_max - z_min)
                
        # --- EJE Z ---
        if (x_max > x_min) and (y_max > y_min) and (s == 0.0):
            if b['Zmax'] >= bi['Zmin'] - p * b_H:
                s = (x_max - x_min) * (y_max - y_min)
            elif bi['Zmax'] >= b['Zmin'] - p * b_H:
                s = (x_max - x_min) * (y_max - y_min)
                
        surface += s

    # 4. Evaluación contra las paredes del contenedor
    if b['Xmin'] <= p * b_L:
        surface += (b_W * b_H)
    if b['Xmax'] >= cont_L - p * b_L:
        surface += (b_W * b_H)
        
    if b['Ymin'] <= p * b_W:
        surface += (b_L * b_H)
    if b['Ymax'] >= cont_W - p * b_W:
        surface += (b_L * b_H)
        
    if b['Zmin'] <= p * b_H:
        surface += (b_L * b_W)
    if b['Zmax'] >= cont_H - p * b_H:
        surface += (b_L * b_W)

    # 5. Normalización final basada en bb.getSurface()
    bb_surface = 2.0 * (b_W * b_H + b_L * b_H + b_L * b_W)
    cs_calculado = surface / bb_surface

    diff = abs(target_cs - cs_calculado)
    status = "✅ MATCH PERFECTO" if diff < 1e-4 else "❌ DISCREPANCIA"
    
    print("\n" + "="*60)
    print(f" Sincronización Real Corregida por Desacople de Ejes")
    print("-"*60)
    print(f" Target CS (C++)     : {target_cs:.8f}")
    print(f" CS Calculado (Py)   : {cs_calculado:.8f}")
    print(f" Estado Sincro       : {status} (Diff: {diff:.8f})")
    print("="*60 + "\n")
'''
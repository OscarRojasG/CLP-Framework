import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
import re
import pickle
import numpy as np
from typing import Dict, List, Tuple
from . import settings



class BlockData:
    """Información de un bloque."""
    def __init__(self, block_id: int, metrics: List[float]):
        self.block_id = block_id
        self.metrics = metrics

    def __repr__(self):
        return f"Block(id={self.block_id})"

class ActionData:
    """Acción que consiste en colocar un bloque con ciertas métricas."""
    def __init__(self, block: BlockData, metrics: List[float]):
        self.block = block          # BlockData
        self.metrics = metrics      # métricas asociadas a la acción

    def __repr__(self):
        return f"Action(block={self.block.block_id}, metrics={self.metrics})"
    
class PlacedBlock:
    def __init__(self, block: BlockData, coords: Tuple[int, int, int]):
        self.block = block
        self.coords = coords

class StateData:
    """Estado del entorno: conjunto de acciones posibles y la acción elegida."""
    def __init__(self, actions: List[ActionData], chosen_action: ActionData, placed: List[PlacedBlock]):            
        self.actions = actions              
        self.chosen_action = chosen_action
        self.placed = placed   

    def __repr__(self):
        chosen = self.chosen_action.block.block_id if self.chosen_action else None
        return f"State(coords={self.coords}, actions={len(self.actions)}, chosen={chosen})"



def run_instance(file_path, i, w, base_folder=None):
    """Ejecuta BSG_CLP para una instancia y guarda la salida en un archivo .out dentro de una carpeta específica o en la misma ruta que file_path"""

    # Asegurarse de que la carpeta de salida exista
    # Determinar carpeta destino: si se pasa base_folder, guardamos dentro de output_folder/base_folder
    if base_folder:
        dest_dir = os.path.join(settings.output_folder_path, base_folder)
    else:
        dest_dir = settings.output_folder_path

    os.makedirs(dest_dir, exist_ok=True)

    # Obtener el nombre base de file_path sin la extensión
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # Generar el nombre del archivo de salida con 'i' y extensión .out dentro de dest_dir
    output_file_path = os.path.join(dest_dir, f"{base_filename}-{i}.out")

    # Ejecutar el proceso y capturar la salida
    proc = subprocess.run(
        [settings.bsg_solver_path, settings.instance_folder_path+file_path, "-i", str(i), "-w", str(w), f"--verbose2={str(w*w)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        text=True
    )

    # Guardar la salida en el archivo de salida
    with open(output_file_path, 'w') as f:
        f.write(proc.stdout)


def run_file_instances_parallel(file_path, w=8, max_workers=None):
    # Leer número de instancias
    with open(settings.instance_folder_path+file_path, "r") as f:
        num_instances = int(f.readline().strip())

    # Preparar el nombre de la carpeta base para pasar a run_instance (si se desea agrupar salidas)
    base_folder = os.path.splitext(os.path.basename(file_path))[0] + ".out"
    # Nos aseguramos de que la carpeta output principal exista (las subcarpetas se crearán en run_instance)
    os.makedirs(settings.output_folder_path, exist_ok=True)

    # Ejecutar las instancias en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(num_instances):
            executor.submit(run_instance, file_path, i, w, base_folder)

def parse_blocks(filepath: str) -> dict[int, BlockData]:
    """Lee un archivo con formato de bloques y devuelve {block_id: BlockData}."""
    blocks_info = {}
    start_reading = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Esperar hasta encontrar "BLOCKS"
            if not start_reading:
                if line == "BLOCKS":
                    start_reading = True
                continue

            # Terminar si hay una línea vacía
            if not line:
                break

            parts = line.split()
            block_id = int(parts[0])
            metrics = list(map(float, parts[1:]))
            blocks_info[block_id] = BlockData(block_id, metrics)

    return blocks_info

def parse_states(filepath: str, blocks_info: Dict[int, BlockData]) -> List[StateData]:
    results = []

    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    n = len(lines)
    while i < n:
        if lines[i] == "Actions":
            i += 1
            actions_raw = []

            # --- Actions ---
            while i < n and lines[i] != "Placed":
                parts = lines[i].split()
                act_id = int(parts[0])
                metrics = list(map(float, parts[1:]))
                actions_raw.append((act_id, metrics))
                i += 1

            # --- Placed (puede estar vacío) ---
            placed_blocks = []
            if i < n and lines[i] == "Placed":
                i += 1
                while i < n and lines[i] != "Selected Block":
                    parts = lines[i].split()
                    block_id = int(parts[0])
                    coords = tuple(map(float, parts[1:4]))
                    placed_blocks.append(PlacedBlock(blocks_info[block_id], coords))
                    i += 1

            # --- Selected Block ---
            i += 1  # saltar "Selected Block"
            chosen_block_id = int(lines[i])
            i += 1

            # --- Crear objetos ---
            actions = [
                ActionData(blocks_info[act_id], metrics)
                for act_id, metrics in actions_raw
            ]
            chosen_action = next(
                a for a in actions if a.block.block_id == chosen_block_id
            )

            results.append(StateData(actions, chosen_action, placed_blocks))

        else:
            i += 1

    return results

def get_w(filename: str) -> int:
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if "Beam width:" in line:
                # Extraer la parte después de los dos puntos y convertir a int
                try:
                    return int(line.split(":")[1].strip())
                except ValueError:
                    raise ValueError(f"No se pudo convertir el beam width a entero en línea: {line.strip()}")

    raise ValueError(f"No se encontró 'Beam width:' en el archivo {filename}")


def generate_train_data(filename: str, min_blocks=10000, min_actions=64):
    """
    Genera:
        X_src_all     : (repetido) métricas de todos los bloques
        X_tgt_all     : métricas de las acciones factibles por estado
        Y_all         : vector one-hot del bloque elegido
        placed_all    : vector (min_actions) con índices+1 de bloques colocados (0 = padding)
        coords_masked : matriz (min_actions, 3) con coordenadas de los bloques colocados
    """
    # --- Cargar datos ---
    blocks_info = parse_blocks(filename)
    if len(blocks_info) < min_blocks:
        return [], [], [], [], []

    states = parse_states(filename, blocks_info)

    # --- Mapa global de índices (basado en X_src) ---
    block_ids = list(blocks_info.keys())
    global_index_map = {block_id: idx for idx, block_id in enumerate(block_ids)}

    # --- X_src fijo ---
    X_src = np.array(
        [blocks_info[b_id].metrics for b_id in block_ids],
        dtype=float
    )

    # --- Inicialización ---
    X_src_all, X_tgt_all, Y_all, placed_all, coords_masked = [], [], [], [], []

    # --- Recorremos los estados ---
    for h, state in enumerate(states):
        # --- Acciones disponibles ---
        # [1:] Ignoramos eval (VCS)
        X_tgt = np.array([[global_index_map[action.block.block_id]] + action.metrics[1:] for action in state.actions], dtype=float)

        # --- Validaciones ---
        if len(X_tgt) < min_actions:
            continue
        if not np.isfinite(X_tgt).all():
            continue

        # --- Etiqueta one-hot ---
        num_actions = len(state.actions)
        Y = np.zeros(num_actions, dtype=float)
        for i, action in enumerate(state.actions):
            if action.block.block_id == state.chosen_action.block.block_id:
                Y[i] = 1.0
                break

        # --- Agregar estado ---
        X_src_all.append(X_src)
        X_tgt_all.append(X_tgt)
        Y_all.append(Y)

        # --- Vector placed (bloques ya colocados) ---
        placed = np.full(min_actions, -1, dtype=int)
        for idx, placed_block in enumerate(state.placed):
            placed[idx] = global_index_map[placed_block.block.block_id]  # índice directo en X_src
        placed_all.append(placed)

        # --- coords_masked: coordenadas relativas alineadas con placed ---
        coords_mat = np.zeros((min_actions, 3), dtype=float)
        for idx, placed_block in enumerate(state.placed):
            coords_mat[idx] = placed_block.coords

        coords_masked.append(coords_mat)

    return (
        np.array(X_src_all, dtype=np.float32),
        np.array(X_tgt_all, dtype=np.float32),
        np.array(Y_all, dtype=np.float32),
        np.array(placed_all, dtype=np.int32),
        np.array(coords_masked, dtype=np.float32)
    )

def generate_data_from_folder(folder_path):
    all_X_src = []      # Entradas del encoder (bloques estáticos)
    all_X_tgt = []      # Entradas del decoder (acciones por estado)
    all_Y = []          # Etiquetas one-hot
    all_placed = []     # Índices de bloques colocados
    all_coords = []     # Coordenadas de bloques colocados

    # Iterar sobre todos los archivos en la carpeta
    for filename in os.listdir(settings.output_folder_path + folder_path):
        file_path = os.path.join(settings.output_folder_path + folder_path, filename)

        if os.path.isfile(file_path):  # Solo procesar archivos (no directorios)
            # Generar los datos de entrenamiento
            X_src, X_tgt, Y, placed, coords = generate_train_data(file_path)

            # Agregar los datos del archivo actual a las listas generales
            all_X_src.extend(X_src)
            all_X_tgt.extend(X_tgt)
            all_Y.extend(Y)
            all_placed.extend(placed)
            all_coords.extend(coords)

    # Definir el nombre del archivo de salida basado en el nombre de la carpeta
    folder_name = os.path.basename(folder_path)
    output_filename = folder_name.split('.')[0] + ".data"
    output_path = settings.data_folder_path + output_filename

    # Guardar los datos (ahora con placed y coords)
    with open(output_path, "wb") as f:
        pickle.dump({
            "X_src": all_X_src,
            "X_tgt": all_X_tgt,
            "Y": all_Y,
            "placed": all_placed,
            "coords": all_coords
        }, f)

    print(f"Datos guardados en: {output_path}")

def load_data_from_file(filename):
    file_path = settings.data_folder_path + filename
    
    # Abrir el archivo .data y cargar los datos
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Extraer X, Y y block_ids
    X_src = data["X_src"]
    X_tgt = data["X_tgt"]
    Y = data["Y"]
    placed = data["placed"]
    coords = data["coords"]

    return X_src, X_tgt, Y, placed, coords

def join_data_files(filenames, output_filename):
    merged_data = {}  # Diccionario donde se unirán todos los datos

    for filename in filenames:
        file_path = settings.data_folder_path + filename

        # Abrir el archivo .data y cargar los datos
        with open(file_path, "rb") as f:
            data = pickle.load(f)

            # Unir listas de cada clave
            for key, value in data.items():
                if key in merged_data:
                    merged_data[key].extend(value)  # concatenar listas
                else:
                    merged_data[key] = value.copy()  # crear nueva entrada
                    
    # Crear el nombre del archivo de salida
    output_path = settings.data_folder_path + output_filename

    # Guardar el diccionario combinado
    with open(output_path, "wb") as f:
        pickle.dump(merged_data, f)

    print(f"Datos combinados guardados en: {output_path}")
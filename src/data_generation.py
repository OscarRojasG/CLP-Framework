import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
import pickle
import numpy as np
from typing import Dict, List
from settings import INSTANCE_FOLDER, OUTPUT_FOLDER, DATA_FOLDER, BSG_SOLVER_PATH



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
    def __init__(self, block: BlockData, features: List[float]):
        self.block = block
        self.features = features

class StateData:
    """Estado del entorno: conjunto de acciones posibles y la acción elegida."""
    def __init__(self, actions: List[ActionData], chosen_action: ActionData, placed: List[PlacedBlock]):            
        self.actions = actions              
        self.chosen_action = chosen_action
        self.placed = placed   

    def __repr__(self):
        chosen = self.chosen_action.block.block_id if self.chosen_action else None
        return f"State(coords={self.coords}, actions={len(self.actions)}, chosen={chosen})"



def run_instance(instance_filename, i, w, base_folder=None):
    """Ejecuta BSG_CLP para una instancia y guarda la salida en un archivo .out dentro de una carpeta específica o en la misma ruta que file_path"""

    # Asegurarse de que la carpeta de salida exista
    # Determinar carpeta destino: si se pasa base_folder, guardamos dentro de output_folder/base_folder
    if base_folder:
        dest_dir = os.path.join(OUTPUT_FOLDER, base_folder)
    else:
        dest_dir = OUTPUT_FOLDER

    os.makedirs(dest_dir, exist_ok=True)

    # Obtener el nombre base de file_path sin la extensión
    base_filename = os.path.splitext(os.path.basename(instance_filename))[0]

    # Generar el nombre del archivo de salida con 'i' y extensión .out dentro de dest_dir
    output_file_path = os.path.join(dest_dir, f"{base_filename}-{i}.out")

    # Ejecutar el proceso y capturar la salida
    proc = subprocess.run(
        [BSG_SOLVER_PATH, os.path.join(INSTANCE_FOLDER, instance_filename), "-i", str(i), "-w", str(w), f"--verbose2={str(w*w)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        text=True
    )

    # Guardar la salida en el archivo de salida
    with open(output_file_path, 'w') as f:
        f.write(proc.stdout)


def run_instances_parallel(instance_filename, w=8, max_workers=None):
    # Leer número de instancias
    with open(INSTANCE_FOLDER / instance_filename, "r") as f:
        num_instances = int(f.readline().strip())

    # Preparar el nombre de la carpeta base para pasar a run_instance (si se desea agrupar salidas)
    base_folder = os.path.splitext(os.path.basename(instance_filename))[0]
    # Nos aseguramos de que la carpeta output principal exista (las subcarpetas se crearán en run_instance)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Ejecutar las instancias en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(num_instances):
            executor.submit(run_instance, instance_filename, i, w, base_folder)

    print(f"Salida guardada en: {OUTPUT_FOLDER / base_folder}")


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
                    features = list(map(float, parts[1:]))
                    placed_blocks.append(PlacedBlock(blocks_info[block_id], features))
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
    # --- Cargar datos ---
    blocks_info = parse_blocks(filename)
    if len(blocks_info) < min_blocks:
        return [], [], [], [], [], []

    states = parse_states(filename, blocks_info)

    # --- Mapa global de índices (basado en X_src) ---
    block_ids = list(blocks_info.keys())
    global_index_map = {block_id: idx for idx, block_id in enumerate(block_ids)}

    # --- bloques fijos ---
    block_features = np.array(
        [blocks_info[b_id].metrics for b_id in block_ids],
        dtype=float
    )

    block_features_all, action_features_all, placed_features_all, action_blocks_all, placed_blocks_all, Y_all = [], [], [], [], [], []

    # --- Recorremos los estados ---
    for state in states:
        # --- Acciones disponibles ---
        # [1:] Ignoramos eval (VCS)
        action_features = np.array([action.metrics[1:] for action in state.actions], dtype=float)
        action_blocks = np.array([global_index_map[action.block.block_id] for action in state.actions], dtype=int)

        # --- Validaciones ---
        if len(action_features) < min_actions:
            continue
        if not np.isfinite(action_features).all():
            continue

        # --- Etiqueta one-hot ---
        num_actions = len(state.actions)
        Y = np.zeros(num_actions, dtype=int)
        for i, action in enumerate(state.actions):
            if action.block.block_id == state.chosen_action.block.block_id:
                Y[i] = 1
                break

        # --- Bloques colocados ---
        placed_features = np.array([placed.features for placed in state.placed], dtype=float).reshape(-1, 3)
        placed_blocks = np.array([global_index_map[placed.block.block_id] for placed in state.placed], dtype=int)

        n = placed_features.shape[0]
        pad_len = max(0, min_actions - n)

        if pad_len > 0:
            placed_features = np.pad(placed_features, pad_width=((0, pad_len), (0, 0)), mode='constant', constant_values=-1)
            placed_blocks = np.pad(placed_blocks, pad_width=(0, pad_len), mode='constant', constant_values=-1)

        # --- Agregar estado ---
        block_features_all.append(block_features)
        action_features_all.append(action_features)
        action_blocks_all.append(action_blocks)
        placed_features_all.append(placed_features)
        placed_blocks_all.append(placed_blocks)
        Y_all.append(Y)

    return (
        np.array(block_features_all, dtype=np.float32),
        np.array(action_blocks_all, dtype=np.int32),
        np.array(action_features_all, dtype=np.float32),
        np.array(placed_blocks_all, dtype=np.int32),
        np.array(placed_features_all, dtype=np.float32),
        np.array(Y_all, dtype=np.float32)
    )


def generate_data_from_folder(folder_path):
    block_features_all, action_features_all, placed_features_all, action_blocks_all, placed_blocks_all, Y_all = [], [], [], [], [], []

    # Iterar sobre todos los archivos en la carpeta
    for filename in os.listdir(OUTPUT_FOLDER / folder_path):
        file_path = os.path.join(OUTPUT_FOLDER / folder_path, filename)

        if os.path.isfile(file_path):  # Solo procesar archivos (no directorios)
            # Generar los datos de entrenamiento
            block_features, action_blocks, action_features, placed_blocks, placed_features, Y = generate_train_data(file_path)

            # Agregar los datos del archivo actual a las listas generales
            block_features_all.extend(block_features)
            action_features_all.extend(action_features)
            placed_features_all.extend(placed_features)
            action_blocks_all.extend(action_blocks)
            placed_blocks_all.extend(placed_blocks)
            Y_all.extend(Y)

    # Definir el nombre del archivo de salida basado en el nombre de la carpeta
    folder_name = os.path.basename(folder_path)
    output_filename = folder_name.split('.')[0] + ".data"
    output_path = DATA_FOLDER / output_filename

    # Guardar los datos
    with open(output_path, "wb") as f:
        pickle.dump({
            "block_features": np.array(block_features_all, dtype=np.float32),
            "action_blocks": np.array(action_blocks_all, dtype=np.int32),
            "action_features": np.array(action_features_all, dtype=np.float32),
            "placed_blocks": np.array(placed_blocks_all, dtype=np.int32),
            "placed_features": np.array(placed_features_all, dtype=np.float32),
            "Y": np.array(Y_all, dtype=np.int32)
        }, f)

    print(f"Datos guardados en: {output_path}")


def load_data(filename):
    file_path = DATA_FOLDER / filename
    
    # Abrir el archivo .data y cargar los datos
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Extraer datos
    block_features = data["block_features"]
    action_blocks = data["action_blocks"]
    action_features = data["action_features"]
    placed_blocks = data["placed_blocks"]
    placed_features = data["placed_features"]
    Y = data["Y"]

    return block_features, action_blocks, action_features, placed_blocks, placed_features, Y
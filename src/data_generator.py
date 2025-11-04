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


class StateData:
    """Estado del entorno: conjunto de acciones posibles y la acción elegida."""
    def __init__(self, coords: Tuple[int, int, int], actions: List[ActionData], chosen_action: ActionData = None):
        self.coords = coords                # (x, y, z)
        self.actions = actions              # lista de Action
        self.chosen_action = chosen_action  # Action elegida (opcional)

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
    """Lee un archivo y devuelve un diccionario {block_id: BlockData}."""
    blocks_info = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Solo procesar líneas con formato "block:XXXX metrics:"
            if line.startswith("block:") and "metrics:" in line:
                try:
                    # Ejemplo: "block:2886 metrics: 0.46 0.34 0.68 ..."
                    parts = line.split("metrics:")
                    block_id_str = parts[0].replace("block:", "").strip()
                    block_id = int(block_id_str)

                    # Convertir los valores numéricos a float
                    metrics = list(map(float, parts[1].strip().split()))

                    # Guardar en el diccionario
                    blocks_info[block_id] = BlockData(block_id, metrics)

                except Exception as e:
                    print(f"Error parsing line: {line} -> {e}")

    return blocks_info

def parse_states(filepath: str, blocks_info: Dict[int, BlockData]) -> List[StateData]:
    """
    Parsea un archivo de acciones y devuelve una lista de objetos State.
    Cada State contiene:
      - coordenadas de inserción
      - lista de posibles acciones (Action)
      - la acción elegida (Action)
    """
    results = []
    current_block = None

    re_selected = re.compile(r"selected block:(\d+)\s+space:\((\d+),(\d+),(\d+)\)")
    re_action = re.compile(r"action block:(\d+)\s+eval:\s+([0-9eE+.\s\-infINF]+)")

    def finalize_block(block_tuple):
        if not block_tuple:
            return

        chosen_block_id, coords, actions_raw = block_tuple
        actions = [
            ActionData(blocks_info[act_id], act_metrics)
            for act_id, act_metrics in actions_raw
            if act_id in blocks_info
        ]

        # Buscar la acción elegida entre las posibles
        chosen_action = next((a for a in actions if a.block.block_id == chosen_block_id), None)
        if chosen_action is None:
            raise ValueError(f"Bloque elegido {chosen_block_id} no aparece entre las acciones disponibles.")

        results.append(StateData(coords, actions, chosen_action))

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m_sel = re_selected.match(line)
            if m_sel:
                finalize_block(current_block)
                block_id = int(m_sel.group(1))
                coords = tuple(map(int, m_sel.groups()[1:]))
                current_block = (block_id, coords, [])
                continue

            m_act = re_action.match(line)
            if m_act and current_block:
                act_id = int(m_act.group(1))
                tokens = m_act.group(2).split()
                nums = []
                for tok in tokens:
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        nums.append(float("nan"))
                current_block[2].append((act_id, nums))

    finalize_block(current_block)
    return results

def parse_actions(filepath: str, blocks_info: Dict[int, BlockData]) -> List[ActionData]:
    """
    Parsea un archivo que contiene únicamente líneas del tipo:
        action block:<id> eval:<valores>

    Devuelve una lista de ActionData, cada uno asociado a su BlockData.
    """
    actions = []
    re_action = re.compile(r"action block:(\d+)\s+eval:\s+([0-9eE+.\s\-infINF]+)")

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = re_action.match(line)
            if not m:
                continue

            act_id = int(m.group(1))
            eval_str = m.group(2)
            tokens = eval_str.split()
            evals = []
            for tok in tokens:
                try:
                    evals.append(float(tok))
                except ValueError:
                    evals.append(float("nan"))

            if act_id in blocks_info:
                actions.append(ActionData(blocks_info[act_id], evals))
            else:
                # Si el block_id no está en blocks_info, lo ignoramos
                print(f"[WARN] Block ID {act_id} no encontrado en blocks_info, se ignora.")

    return actions

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
    placed_indices = []   # índices globales (en X_src)
    coords_all = []       # coordenadas correspondientes a esos bloques

    # --- Recorremos los estados ---
    for state in states:
        chosen_block = state.chosen_action.block
        chosen_id = chosen_block.block_id

        # --- Validar bloque ---
        if chosen_id not in global_index_map:
            continue
        chosen_index = global_index_map[chosen_id]

        # --- Acciones disponibles ---
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
            if action.block.block_id == chosen_id:
                Y[i] = 1.0
                break

        # --- Agregar estado ---
        X_src_all.append(X_src)
        X_tgt_all.append(X_tgt)
        Y_all.append(Y)

        # --- Vector placed (bloques ya colocados) ---
        placed = np.full(min_actions, -1, dtype=int)
        for idx, block_idx in enumerate(placed_indices[:min_actions]):
            placed[idx] = block_idx  # índice directo en X_src
        placed_all.append(placed)

        # --- coords_masked: coordenadas relativas alineadas con placed ---
        coords_mat = np.zeros((min_actions, 3), dtype=float)
        current_coords = np.array(state.coords, dtype=float)

        for idx, block_idx in enumerate(placed_indices[:min_actions]):
            # Coordenadas relativas = bloque anterior - bloque actual
            coords_mat[idx] = coords_all[idx] - current_coords

        coords_masked.append(coords_mat)

        # Registrar el bloque actual como colocado
        coords = np.array(state.coords, dtype=float)
        placed_indices.append(chosen_index)
        coords_all.append(coords)

    return X_src_all, X_tgt_all, Y_all, placed_all, coords_masked

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
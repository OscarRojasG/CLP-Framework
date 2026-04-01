import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
import numpy as np
from settings import INSTANCE_FOLDER, OUTPUT_FOLDER, DATA_FOLDER, BSG_SOLVER_PATH
from misc.labels import LabelType


def run_instance(instance_filename, i, w, base_folder, min_fr, num_actions, double_effort):
    """Ejecuta BSG_CLP para una instancia y guarda la salida en un archivo .out dentro de una carpeta específica o en la misma ruta que file_path"""

    # Asegurarse de que la carpeta de salida exista
    # Determinar carpeta destino: si se pasa base_folder, guardamos dentro de output_folder/base_folder
    if base_folder:
        dest_dir = os.path.join(OUTPUT_FOLDER, base_folder)
    else:
        dest_dir = OUTPUT_FOLDER

    if not num_actions:
        num_actions = w * w

    os.makedirs(dest_dir, exist_ok=True)

    # Obtener el nombre base de file_path sin la extensión
    base_filename = os.path.splitext(os.path.basename(instance_filename))[0]

    # Generar el nombre del archivo de salida con 'i' y extensión .out dentro de dest_dir
    output_file_path = os.path.join(dest_dir, f"{base_filename}-{i}.out")

    # Ejecutar el proceso y capturar la salida
    with open(output_file_path, 'w') as f:
        cmd = [
            BSG_SOLVER_PATH, 
            os.path.join(INSTANCE_FOLDER, instance_filename), 
            "-i", str(i), 
            "-w", str(w), 
            f"--min_fr={min_fr}", 
            f"--verbose2={num_actions}"
        ]

        if double_effort:
            cmd.append("--de")

        subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )

def run_instances_parallel(instance_filename, w=8, max_workers=None, min_fr=1, num_actions=None, double_effort=True, base_folder=None):
    # Leer número de instancias
    with open(INSTANCE_FOLDER / instance_filename, "r") as f:
        num_instances = int(f.readline().strip())

    # Preparar el nombre de la carpeta base para pasar a run_instance (si se desea agrupar salidas)
    if not base_folder:
        base_folder = os.path.splitext(os.path.basename(instance_filename))[0]
    # Nos aseguramos de que la carpeta output principal exista (las subcarpetas se crearán en run_instance)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not num_actions:
        num_actions = w * w

    # Ejecutar las instancias en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_instance, instance_filename, i, w, base_folder, min_fr, num_actions, double_effort) 
            for i in range(num_instances)
        ]
            
        completed_count = 0
        for _ in as_completed(futures):
            completed_count += 1
            percentage = (completed_count / num_instances) * 100
            print(f"\rProgreso: {percentage:.2f}% ({completed_count}/{num_instances})", end="")

    print(f"\nSalida guardada en: {OUTPUT_FOLDER / base_folder}")


def read_output(filepath: str):
    block_features = []
    action_blocks_all = []
    action_features_all = []
    placed_blocks_all = []
    placed_features_all = []
    space_features_all = []
    selected_blocks_all = []
    vcs_evals_all = []
    
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    i = 0
    n = len(lines)
    while i < n:
        if lines[i] == "BLOCKS":
            i += 1
            while i < n and lines[i] != "SOLVE STEPS":
                parts = lines[i].split()
                metrics = list(map(float, parts))
                block_features.append(metrics)
                i += 1
        
        if lines[i] == "Actions":
            i += 1
            action_blocks = []
            action_features = []
            vcs_evals = []
            while i < n and lines[i] != "Placed":
                parts = lines[i].split()
                act_block = int(parts[0])
                act_feat = list(map(float, parts[2:]))
                vcs_eval = float(parts[1])
                action_blocks.append(act_block)
                action_features.append(act_feat)
                vcs_evals.append(vcs_eval)
                i += 1
            action_blocks_all.append(action_blocks)
            action_features_all.append(action_features)
            vcs_evals_all.append(vcs_evals)

        if lines[i] == "Placed":
            i += 1
            placed_blocks = []
            placed_features = []
            while i < n and lines[i] != "Space":
                parts = lines[i].split()
                pl_block = int(parts[0])
                pl_feat = list(map(float, parts[1:]))
                placed_blocks.append(pl_block)
                placed_features.append(pl_feat)
                i += 1
            placed_blocks_all.append(placed_blocks)
            placed_features_all.append(placed_features)
                
        if lines[i] == "Space":
            i += 1
            sp_feat = list(map(float, lines[i].split()))
            space_features_all.append(sp_feat)
            
        if lines[i] == "Selected Block":
            i += 1
            selected_block = int(lines[i])
            selected_blocks_all.append(selected_block)
            
        i += 1
        
    return block_features, action_blocks_all, action_features_all, placed_blocks_all, placed_features_all, space_features_all, selected_blocks_all, vcs_evals_all

def generate_train_data(filename: str, min_actions=64, padding_blocks=10000, padding_placed=64):
    block_features, action_blocks_all, action_features_all, placed_blocks_all, placed_features_all, space_features_all, selected_blocks_all, vcs_evals_all = read_output(filename)
    num_states = len(action_features_all)
    
    # Padding bloques
    block_features = np.array(block_features, dtype=float)    
    n = block_features.shape[0]
    pad_len = max(0, padding_blocks - n)
    if pad_len > 0:
        block_features = np.pad(block_features, pad_width=((0, pad_len), (0, 0)), mode='constant', constant_values=-1)
        
    np_block_features = []
    np_action_blocks = []
    np_action_features = []
    np_placed_blocks = []
    np_placed_features = []
    np_space_features = []
    np_Y = []
    
    for i in range(num_states):
        # Truncar acciones
        if len(action_features_all[i]) < min_actions:
            continue
        
        # Etiqueta
        Y = np.zeros(min_actions, dtype=float)

        action_blocks_all[i] = action_blocks_all[i][:min_actions]
        action_features_all[i] = action_features_all[i][:min_actions]

        found = False    
        for j, action_block in enumerate(action_blocks_all[i]):
            if action_block == selected_blocks_all[i]:
                Y[j] = 1
                found = True
                break

        if not found:
            continue
            
        # Padding bloques colocados
        placed_blocks = np.array(placed_blocks_all[i], dtype=int)
        placed_features = np.array(placed_features_all[i], dtype=float).reshape(-1, 4)
        
        n = placed_blocks.shape[0]
        pad_len = max(0, padding_placed - n)

        if pad_len > 0:
            placed_blocks = np.pad(placed_blocks, pad_width=(0, pad_len), mode='constant', constant_values=-1)
            placed_features = np.pad(placed_features, pad_width=((0, pad_len), (0, 0)), mode='constant', constant_values=-1)
    
        # Añadir a listas
        np_block_features.append(block_features)
        np_action_blocks.append(np.array(action_blocks_all[i], dtype=int))
        np_action_features.append(np.array(action_features_all[i], dtype=float))
        np_placed_blocks.append(placed_blocks)  
        np_placed_features.append(placed_features)
        np_space_features.append(np.array(space_features_all[i], dtype=float))  
        np_Y.append(Y)
    
    return (
        np.array(np_block_features, dtype=np.float32),
        np.array(np_action_blocks, dtype=np.int32),
        np.array(np_action_features, dtype=np.float32),
        np.array(np_placed_blocks, dtype=np.int32),
        np.array(np_placed_features, dtype=np.float32),
        np.array(np_space_features, dtype=np.float32),
        np.array(np_Y, dtype=np.float32),
    )

def generate_data_from_folder(folder_path, output_filename=None, label_type=LabelType.BEST_ACTION, min_actions=64):
    block_features_all, action_features_all, placed_features_all, action_blocks_all, placed_blocks_all, space_features_all, Y_all = [], [], [], [], [], [], []
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # Iterar sobre todos los archivos en la carpeta
    for input_filename in os.listdir(OUTPUT_FOLDER / folder_path):
        file_path = os.path.join(OUTPUT_FOLDER / folder_path, input_filename)

        if os.path.isfile(file_path):  # Solo procesar archivos (no directorios)
            # Generar los datos de entrenamiento
            block_features, action_blocks, action_features, placed_blocks, placed_features, space_features, Y = generate_train_data(file_path, min_actions)

            # Agregar los datos del archivo actual a las listas generales
            block_features_all.extend(block_features)
            action_features_all.extend(action_features)
            placed_features_all.extend(placed_features)
            action_blocks_all.extend(action_blocks)
            placed_blocks_all.extend(placed_blocks)
            space_features_all.extend(space_features)
            Y_all.extend(Y)

    # Definir el nombre del archivo de salida basado en el nombre de la carpeta
    if output_filename is None:
        folder_name = os.path.basename(folder_path)
        output_filename = folder_name.split('.')[0]

    output_path = DATA_FOLDER / (output_filename + ".data")

    # Guardar los datos
    with h5py.File(output_path, "w") as f:
        f.create_dataset("block_features", data=np.array(block_features_all, dtype=np.float32))
        f.create_dataset("action_blocks", data=np.array(action_blocks_all, dtype=np.int32))
        f.create_dataset("action_features", data=np.array(action_features_all, dtype=np.float32))
        f.create_dataset("placed_blocks", data=np.array(placed_blocks_all, dtype=np.int32))
        f.create_dataset("placed_features", data=np.array(placed_features_all, dtype=np.float32))
        f.create_dataset("space_features", data=np.array(space_features_all, dtype=np.float32))
        Y = f.create_dataset("Y", data=np.array(Y_all, dtype=np.float32))
        Y.attrs["label_type"] = label_type.value

    print(f"Datos guardados en: {output_path}")
import subprocess
import os
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
from settings import INSTANCE_FOLDER, OUTPUT_FOLDER, DATA_FOLDER, BSG_SOLVER_PATH
from data.objects import *
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def read_output(filepath: str):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    it = iter(lines)
    blocks = []

    # 1. Leer los BLOCKS una sola vez al principio
    for line in it:
        if line == "BLOCKS":
            for l in it:
                if l == "SOLVE STEPS":
                    break
                blocks.append(Block(list(map(float, l.split()))))
            break # Salimos del guardado inicial de bloques

    # 2. Bucle principal para el resto del archivo (Actions, Placed, Space...)
    for line in it:
        # Buscamos el inicio de una iteración (Actions)
        if line == "Actions":
            actions = []
            pblocks = []
            space = None

            # --- Leer Actions ---
            for l in it:
                if l == "Placed":
                    break
                actions.append(Action(list(map(float, l.split()))))

            # --- Leer Placed ---
            for l in it:
                if l == "Space":
                    break
                pblocks.append(PBlock(list(map(float, l.split()))))
                
            # --- Leer Space ---
            # La siguiente línea contiene directamente los datos de Space
            space = Space(list(map(float, next(it).split())))
            
            # --- Acción elegida ---
            next(it)
            selected_block = int(next(it))

            # --- Volumen actual ---
            next(it)
            volume = float(next(it))

            # --- Dato final (Greedy) ---
            next(it)
            greedy = float(next(it))
            
            # Retornamos los datos actuales JUNTO con los bloques iniciales
            yield blocks, actions, pblocks, space, selected_block, greedy


def get_rank(actions, selected_block):
    for i, action in enumerate(actions):
        if action.block_id == selected_block:
            return i
        

def generate_data_from_file(filepath, input_adapter, output_adapter, ranks, min_blocks, min_actions):
    # read_output(filepath) ahora es un objeto generador
    for blocks, actions, pblocks, space, selected_block, greedy in read_output(filepath):
        if len(blocks) < min_blocks:
            return
        
        if len(actions) < min_actions:
            continue

        input_data = input_adapter.input_2_vec(blocks, space, pblocks, actions)
        input_adapter.add(input_data)

        output_data = output_adapter.output_2_vec(actions, selected_block, greedy)
        output_adapter.add(output_data)

        rank = get_rank(actions, selected_block)
        ranks.append(rank)


def process_single_file(filepath, input_adapter, output_adapter, min_blocks, min_actions):
    """
    Procesa un solo archivo de forma aislada y acumula los resultados locales.
    No toca ningún estado global.
    """
    local_inputs = []
    local_outputs = []
    local_ranks = []
    
    for blocks, actions, pblocks, space, selected_block, greedy in read_output(filepath):
        if len(blocks) < min_blocks:
            # Si un archivo no cumple la condición de bloques, 
            # descartamos lo que llevamos de este archivo.
            return [], [], []
        
        if len(actions) < min_actions:
            continue

        # Nota: Aquí asumimos que input_adapter.input_2_vec y output_2_vec 
        # son funciones puras (no modifican estado interno del adaptador)
        input_data = input_adapter.input_2_vec(blocks, space, pblocks, actions)
        local_inputs.append(input_data)

        output_data = output_adapter.output_2_vec(actions, selected_block, greedy)
        local_outputs.append(output_data)

        rank = get_rank(actions, selected_block)
        local_ranks.append(rank)
        
    return local_inputs, local_outputs, local_ranks


def generate_data(folder, input_adapter, output_adapter, min_blocks, min_actions, prefix=None):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # 1. Obtener y ordenar la lista de archivos para asegurar determinismo
    folder_path = OUTPUT_FOLDER / folder
    filenames = sorted(os.listdir(folder_path)) 
    filepaths = [os.path.join(folder_path, fname) for fname in filenames]

    all_inputs = []
    all_outputs = []
    ranks = []

    task = partial(
        process_single_file, 
        input_adapter=input_adapter, 
        output_adapter=output_adapter, 
        min_blocks=min_blocks, 
        min_actions=min_actions
    )

    # 2. Procesar en paralelo usando un pool de hilos
    # Usamos max_workers=None para que Python elija según los cores, o puedes fijar un número.
    with ThreadPoolExecutor() as executor:
        results = executor.map(task, filepaths)
        
        # 3. Reunir los resultados secuencialmente en el hilo principal
        for local_inputs, local_outputs, local_ranks in results:
            all_inputs.extend(local_inputs)
            all_outputs.extend(local_outputs)
            ranks.extend(local_ranks)

    # 4. Insertar de golpe (o secuencialmente) en los adaptadores finales
    for inp in all_inputs:
        input_adapter.add(inp)
        
    for out in all_outputs:
        output_adapter.add(out)

    input_data = input_adapter.get()
    output_data = output_adapter.get()

    output_name = f"{prefix}_{folder}.data" if prefix else f"{folder}.data"
    save_data(input_data, output_data, ranks, output_name)


def save_data(input_data, output_data, ranks, output_name):
    output_path = DATA_FOLDER / f"{output_name}"

    with h5py.File(output_path, "w") as f:
        g_input = f.create_group("input")
        g_output = f.create_group("output")

        input_keys = list(input_data.keys())
        for key in input_keys:
            g_input.create_dataset(key, data=input_data[key])
        g_input.attrs['key_order'] = [k for k in input_keys]

        output_keys = list(output_data.keys())
        for key in output_keys:
            g_output.create_dataset(key, data=output_data[key])
        g_output.attrs['key_order'] = [k for k in output_keys]

        f.create_dataset("ranks", data=np.stack(ranks, dtype=np.int32))

    print(f"Datos guardados en: {output_path} (Tamaño {len(output_data[key])})")

    
def run_instance(instance_filename, i, w, base_folder, min_fr, num_actions, double_effort):
    """Ejecuta BSG_CLP para una instancia y guarda la salida en un archivo .out dentro de una carpeta específica o en la misma ruta que file_path"""

    # Guardamos dentro de output_folder/base_folder
    dest_dir = os.path.join(OUTPUT_FOLDER, base_folder)

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


def run_instances_parallel(instance_filename, w=8, max_workers=None, min_fr=1, double_effort=True):
    # Leer número de instancias
    with open(INSTANCE_FOLDER / instance_filename, "r") as f:
        num_instances = int(f.readline().strip())

    # Preparar el nombre de la carpeta base para pasar a run_instance
    base_folder = os.path.splitext(os.path.basename(instance_filename))[0]

    # Nos aseguramos de que la carpeta output principal exista (las subcarpetas se crearán en run_instance)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Ejecutar las instancias en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_instance, instance_filename, i, w, base_folder, min_fr, w*w, double_effort) 
            for i in range(num_instances)
        ]
            
        completed_count = 0
        for _ in as_completed(futures):
            completed_count += 1
            percentage = (completed_count / num_instances) * 100
            print(f"\rProgreso: {percentage:.2f}% ({completed_count}/{num_instances})", end="")

    print(f"\nSalida guardada en: {OUTPUT_FOLDER / base_folder}")
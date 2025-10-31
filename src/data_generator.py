import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
import re
import pickle
from . import settings


output_folder = "outputs/"
data_folder = "data/"


def run_instance(file_path, i, w, base_folder=None):
    """Ejecuta BSG_CLP para una instancia y guarda la salida en un archivo .out dentro de una carpeta específica o en la misma ruta que file_path"""

    # Asegurarse de que la carpeta de salida exista
    # Determinar carpeta destino: si se pasa base_folder, guardamos dentro de output_folder/base_folder
    if base_folder:
        dest_dir = os.path.join(output_folder, base_folder)
    else:
        dest_dir = output_folder

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
    os.makedirs(output_folder, exist_ok=True)

    # Ejecutar las instancias en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(num_instances):
            executor.submit(run_instance, file_path, i, w, base_folder)

def parse_blocks(filepath):
    blocks_info = []
    container_dims = None  # (L, W, H)

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Detectar dimensiones del contenedor
            if container_dims is None and line.count(" ") == 2 and line.replace(" ", "").isdigit():
                # Ejemplo: "587 233 220"
                parts = line.split()
                container_dims = tuple(map(int, parts))  # (L, W, H)

            # Detectar bloques
            if line.startswith("block:"):
                # Ejemplo: block: 1 (51,66,45)
                try:
                    prefix, dims_str = line.split("(")
                    block_id = int(prefix.split()[1])
                    dims = dims_str.strip(")").split(",")
                    block_dims = tuple(map(int, dims))  # (L, W, H)

                    # Normalizar
                    l_ratio = block_dims[0] / container_dims[0]
                    w_ratio = block_dims[1] / container_dims[1]
                    h_ratio = block_dims[2] / container_dims[2]

                    blocks_info.append([block_id, l_ratio, w_ratio, h_ratio])
                except Exception as e:
                    print(f"Error parsing line: {line} -> {e}")

    # lista [info_b1, info_b2, ...]
    # info bloque: [block_id, l_ratio, w_ratio, h_ratio]
    return blocks_info


def parse_actions(filepath):
    results = []
    current_block = None

    re_selected = re.compile(r"selected block:(\d+)\s+space:\((\d+),(\d+),(\d+)\)")
    re_action = re.compile(r"action block:(\d+)\s+eval:\s+([0-9eE+.\s\-infINF]+)")

    # Abrimos el archivo y leemos su contenido
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # selected block
            m_sel = re_selected.match(line)
            if m_sel:
                if current_block:
                    results.append(current_block)
                block_id = int(m_sel.group(1))
                coords = tuple(map(int, m_sel.groups()[1:]))
                current_block = (block_id, coords, [])
                continue

            # action block
            m_act = re_action.match(line)
            if m_act and current_block:
                act_id = int(m_act.group(1))
                tokens = m_act.group(2).split()
                nums = []
                for tok in tokens:
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        # fallback por si aparece algo inesperado
                        nums.append(float("nan"))
                current_block[2].append((act_id, nums))

    if current_block:
        results.append(current_block)

    # [(bloque_elegido:int, coordenadas:tupla, acciones disponibles:lista)]
    # acción = (bloque_id:int, features:lista)
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



def generate_train_data(filename, min_blocks=10000):
    X_src = []  # Para los datos del encoder (bloques)
    X_tgt = []  # Para los datos del decoder (acciones)
    Y = []      # Para las predicciones (vectores one-hot)
    ids = []    # Para las ids de los bloques

    blocks_data = parse_blocks(filename)
    if len(blocks_data) < min_blocks: return X_src, X_tgt, Y, ids

    actions_data = parse_actions(filename)
    seq_size = get_w(filename)**2

    # Eliminar ID de cada bloque (irrelevante)
    blocks_data = [sub[1:] for sub in blocks_data]

    # Recorremos cada bloque
    for action in actions_data:
        # Extraemos las características del bloque (el tercer elemento en cada bloque)
        features = [entry[1] for entry in action[2]] # Obtiene las características de cada tupla en el bloque

        # Consideramos solo cuando bloques = w2
        if len(features) < seq_size: continue

        # Añadimos las características de este bloque a X
        X_src.append(blocks_data)
        X_tgt.append(features)

        # Extraemos el id del bloque elegido (primer valor de la tupla)
        selected_id = action[0]

        # Crear el vector one-hot para Y
        one_hot = [0] * seq_size  # Inicializa un vector con ceros

        # Encontrar la posición de block_id en las tuplas de cada bloque
        best_block_idx = None
        for i, entry in enumerate(action[2]):
            if entry[0] == selected_id:  # Compara el id de la tupla con el id del bloque
                one_hot[i] = 1  # Marca la posición correspondiente en el vector one-hot
                best_block_idx = i
                break

        # one_hot = 1 para features iguales al mejor bloque
        best_block_features = features[best_block_idx]
        for entry in features:
            if entry == best_block_features:
                idx = features.index(entry)
                one_hot[idx] = 1  

        # Añadir el vector one-hot a Y
        Y.append(one_hot)

        # Añadir las ids del bloque a block_ids
        action_ids = [entry[0] for entry in action[2]]  # Extrae solo el ID de cada tupla
        ids.append(action_ids)

    return X_src, X_tgt, Y, ids


def generate_data_from_folder(folder_path):
    all_X_src = []  # Lista para almacenar entradas del encoder
    all_X_tgt = []  # Lista para almacenar entradas del decoder
    all_Y = []  # Lista para almacenar todos los vectores one-hot
    all_ids = []  # Lista para almacenar todos los block_ids

    # Iterar sobre todos los archivos en la carpeta
    for filename in os.listdir(settings.output_folder_path + folder_path):
        file_path = os.path.join(settings.output_folder_path + folder_path, filename)

        if os.path.isfile(file_path):  # Solo procesar archivos (no directorios)
            # Generar los datos de entrenamiento
            X_src, X_tgt, Y, ids = generate_train_data(file_path)

            # Agregar los datos del archivo actual a las listas generales
            all_X_src.extend(X_src)
            all_X_tgt.extend(X_tgt)
            all_Y.extend(Y)
            all_ids.extend(ids)

    # Definir el nombre del archivo de salida basado en el nombre de la carpeta
    folder_name = os.path.basename(folder_path)  # Obtiene el nombre de la carpeta
    output_filename = folder_name.split('.')[0] + ".data"  # Eliminar la extensión .out si está presente y agregar .data
    output_path = data_folder + output_filename

    # Guardar los datos en el archivo (en este caso, usaremos pickle para guardar en formato binario)
    with open(output_path, "wb") as f:
        pickle.dump({"X_src": all_X_src, "X_tgt": all_X_tgt, "Y": all_Y, "ids": all_ids}, f)

    print(f"Datos guardados en: {output_path}")


def load_data_from_file(filename):
    file_path = f"data/{filename}"
    
    # Abrir el archivo .data y cargar los datos
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Extraer X, Y y block_ids
    X_src = data["X_src"]
    X_tgt = data["X_tgt"]
    Y = data["Y"]
    ids = data["ids"]

    return X_src, X_tgt, Y, ids

def join_data_files(filenames, output_filename):
    merged_data = {}  # Diccionario donde se unirán todos los datos

    for filename in filenames:
        file_path = f"data/{filename}"

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
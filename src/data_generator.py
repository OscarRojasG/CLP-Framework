import subprocess
import os
from concurrent.futures import ProcessPoolExecutor
import re
import pickle

def run_instance(exe_path, file_path, i, w, output_path=None):
    """Ejecuta BSG_CLP para una instancia y guarda la salida en un archivo .out dentro de una carpeta específica o en la misma ruta que file_path"""

    # Si output_path es None, usar la misma carpeta que file_path
    if output_path is None:
        output_path = os.path.dirname(file_path)

    # Obtener el nombre base de file_path sin la extensión
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # Generar el nombre del archivo de salida con 'i' y extensión .out
    output_file_path = os.path.join(output_path, f"{base_filename}-{i}.out")

    # Ejecutar el proceso y capturar la salida
    proc = subprocess.run(
        [exe_path, file_path, "-i", str(i), "-w", str(w), f"--verbose2={str(w*w)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        text=True
    )

    # Guardar la salida en el archivo de salida
    with open(output_file_path, 'w') as f:
        f.write(proc.stdout)


def run_file_instances_parallel(exe_path, file_path, w=8, max_workers=None):
    # Leer número de instancias
    with open(file_path, "r") as f:
        num_instances = int(f.readline().strip())

    # Crear la carpeta de salida con el nombre del archivo (sin la extensión) y añadir ".out" al final
    output_folder = os.path.splitext(os.path.basename(file_path))[0] + ".out"  # Usamos el nombre del archivo sin la extensión y añadimos ".out"
    os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe

    # Ejecutar las instancias en paralelo
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(num_instances):
            executor.submit(run_instance, exe_path, file_path, i, w, output_folder)


def parse_output_file(file_path):
    results = []
    current_block = None

    re_selected = re.compile(r"selected block:(\d+)\s+space:\((\d+),(\d+),(\d+)\)")
    re_action = re.compile(r"action block:(\d+)\s+eval:\s+([0-9eE+.\s\-infINF]+)")

    # Abrimos el archivo y leemos su contenido
    with open(file_path, 'r') as f:
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

    return results


def generate_train_data(data, pad=64):
    X = []  # Para los datos de entrenamiento
    Y = []  # Para las predicciones (vectores one-hot)
    block_ids = []  # Para las ids de los bloques

    # Recorremos cada bloque
    for block in data:
        # Extraemos las características del bloque (el tercer elemento en cada bloque)
        features = [entry[1][:4] for entry in block[2]]  # Obtiene las características de cada tupla en el bloque

        # Rellenamos con ceros si el número de características es menor que pad
        while len(features) < pad:
            features.append([0.0] * len(features[0]))  # Añadir una lista de ceros de la misma longitud que las características

        # Añadimos las características de este bloque a X
        X.append(features[:pad])  # Aseguramos que no se sobrepasen los "pad" elementos

        # Extraemos el id del bloque (primer valor de la tupla)
        block_id = block[0]

        # Crear el vector one-hot para Y
        one_hot = [0] * pad  # Inicializa un vector con ceros

        # Encontrar la posición de block_id en las tuplas de cada bloque
        for i, entry in enumerate(block[2]):
            if entry[0] == block_id:  # Compara el id de la tupla con el id del bloque
                one_hot[i] = 1  # Marca la posición correspondiente en el vector one-hot
                break

        # Rellenamos con ceros si la longitud de one_hot es menor que pad
        while len(one_hot) < pad:
            one_hot.append(0)  # Añadir ceros al final si hace falta

        # Añadir el vector one-hot a Y
        Y.append(one_hot[:pad])  # Aseguramos que no se sobrepasen los "pad" elementos

        # Añadir las ids del bloque a block_ids y rellenar con ceros si es necesario
        ids = [entry[0] for entry in block[2]]  # Extrae solo el ID de cada tupla
        while len(ids) < pad:
            ids.append(0)  # Rellenar con ceros hasta alcanzar "pad"
        block_ids.append(ids[:pad])  # Aseguramos que no se sobrepasen los "pad" elementos

    return X, Y, block_ids


def generate_data_from_folder(folder_path, pad=64):
    all_X = []  # Lista para almacenar todos los datos de entrenamiento
    all_Y = []  # Lista para almacenar todos los vectores one-hot
    all_block_ids = []  # Lista para almacenar todos los block_ids

    # Iterar sobre todos los archivos en la carpeta
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):  # Solo procesar archivos (no directorios)
            # Parsear el archivo
            data = parse_output_file(file_path)

            # Generar los datos de entrenamiento
            X, Y, block_ids = generate_train_data(data, pad=pad)

            # Agregar los datos del archivo actual a las listas generales
            all_X.extend(X)
            all_Y.extend(Y)
            all_block_ids.extend(block_ids)

    # Definir el nombre del archivo de salida basado en el nombre de la carpeta
    folder_name = os.path.basename(folder_path)  # Obtiene el nombre de la carpeta
    output_filename = folder_name.split('.')[0] + ".data"  # Eliminar la extensión .out si está presente y agregar .data
    output_path = os.path.join(os.path.dirname(folder_path), output_filename)  # Guardar el archivo en el nivel superior

    # Guardar los datos en el archivo (en este caso, usaremos pickle para guardar en formato binario)
    with open(output_path, "wb") as f:
        pickle.dump({"X": all_X, "Y": all_Y, "block_ids": all_block_ids}, f)

    print(f"Datos guardados en: {output_path}")
    return output_path


def load_data_from_file(filename):
    file_path = f"data/{filename}"
    
    # Abrir el archivo .data y cargar los datos
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Extraer X, Y y block_ids
    X = data["X"]
    Y = data["Y"]
    block_ids = data["block_ids"]

    return X, Y, block_ids
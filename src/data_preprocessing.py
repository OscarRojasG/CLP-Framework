import os
import pickle
from .data_generator import load_data_from_file

def generate_datasets(filename, cuts):
    """Separa los datos en varios datasets según los cortes y guarda en archivos .data"""

    output_path = "data/datasets"

    # Añadir un "1" al principio del arreglo cuts para asegurar que el primer dataset sea considerado
    cuts = [1] + cuts

    # Asegurarse de que la carpeta de salida exista
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Cargar los datos desde el archivo .data
    print(f"Procesando archivo: {filename}")
    X, Y, block_ids = load_data_from_file(filename)

    # Crear un diccionario para almacenar los datasets
    datasets = {i: {"X": [], "Y": [], "block_ids": []} for i in range(1, len(cuts))}

    # Separar los datos según los cortes
    for i, y_vector in enumerate(Y):
        pos = y_vector.index(1)  # Encontrar la posición del 1 en el vector one-hot

        # Asignar el dato al dataset correspondiente según el corte
        for j in range(len(cuts)-1):
            if pos < cuts[j+1]:
                datasets[j + 1]["X"].append(X[i])
                datasets[j + 1]["Y"].append(Y[i])
                datasets[j + 1]["block_ids"].append(block_ids[i])
                break  # Una vez asignado, salir del loop de cortes

    # Guardar cada dataset en un archivo
    for i, dataset in datasets.items():
        # Generar el nombre del archivo con base en el corte
        start_cut = cuts[i-1] + 1 if i > 1 else 1
        end_cut = cuts[i]  # Cortes entre `cuts[i-1]` y `cuts[i]`

        output_filename = f"{start_cut}-{end_cut}.data"
        output_file_path = os.path.join(output_path, output_filename)

        # Guardar el dataset en el archivo
        with open(output_file_path, "wb") as f:
            pickle.dump({"X": dataset["X"], "Y": dataset["Y"], "block_ids": dataset["block_ids"]}, f)

        print(f"Dataset guardado en: {output_file_path}")


def remove_elements_with_zero(X, Y, blocks_ids):
    # Crear una lista de índices a eliminar
    indices_to_remove = []

    # Recorre blocks_ids y encuentra los índices que contienen al menos un 0
    for idx, arr in enumerate(blocks_ids):
        if 0 in arr:
            indices_to_remove.append(idx)

    # Eliminar los elementos en X, Y y blocks_ids correspondientes a los índices encontrados
    X_filtered = [x for i, x in enumerate(X) if i not in indices_to_remove]
    Y_filtered = [y for i, y in enumerate(Y) if i not in indices_to_remove]
    blocks_ids_filtered = [block for i, block in enumerate(blocks_ids) if i not in indices_to_remove]

    return X_filtered, Y_filtered, blocks_ids_filtered


def remove_elements_with_less_blocks(X, Y, blocks_ids, min_blocks=10000):
    # Crear una lista de índices a eliminar
    indices_to_remove = []

    # Recorre blocks_ids y encuentra los índices que contienen al menos un 0
    for idx, arr in enumerate(X):
        if len(arr[0]) < min_blocks:
            indices_to_remove.append(idx)

    # Eliminar los elementos en X, Y y blocks_ids correspondientes a los índices encontrados
    X_filtered = [x for i, x in enumerate(X) if i not in indices_to_remove]
    Y_filtered = [y for i, y in enumerate(Y) if i not in indices_to_remove]
    blocks_ids_filtered = [block for i, block in enumerate(blocks_ids) if i not in indices_to_remove]

    return X_filtered, Y_filtered, blocks_ids_filtered
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from .data_generator import load_data_from_file

def generate_datasets(filename, cuts, max_size=None):
    """Separa los datos en varios datasets según los cortes y guarda en archivos .data"""

    output_path = "data/datasets"

    # Añadir un "1" al principio del arreglo cuts para asegurar que el primer dataset sea considerado
    cuts = [1] + cuts

    # Asegurarse de que la carpeta de salida exista
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Cargar los datos desde el archivo .data
    print(f"Procesando archivo: {filename}")
    X_src, X_tgt, Y, ids = load_data_from_file(filename)

    # Crear un diccionario para almacenar los datasets
    datasets = {i: {"X_src": [], "X_tgt": [], "Y": [], "ids": []} for i in range(1, len(cuts))}

    # Separar los datos según los cortes
    for i, y_vector in enumerate(Y):
        pos = y_vector.index(1)  # Encontrar la posición del 1 en el vector one-hot

        # Asignar el dato al dataset correspondiente según el corte
        for j in range(len(cuts) - 1):
            if pos < cuts[j + 1]:
                datasets[j + 1]["X_src"].append(X_src[i])
                datasets[j + 1]["X_tgt"].append(X_tgt[i])
                datasets[j + 1]["Y"].append(Y[i])
                datasets[j + 1]["ids"].append(ids[i])
                break  # Una vez asignado, salir del loop de cortes

    # Guardar cada dataset en un archivo
    for i, dataset in datasets.items():
        start_cut = cuts[i - 1] + 1 if i > 1 else 1
        end_cut = cuts[i]
    
        # Si hay un límite de tamaño, truncar el dataset
        if max_size is not None and len(dataset["X_src"]) > max_size:
            for key in dataset:
                dataset[key] = dataset[key][:max_size]

        output_filename = f"{filename.split('.')[0]}_{start_cut}-{end_cut}.data"
        output_file_path = os.path.join(output_path, output_filename)

        # Guardar el dataset en el archivo con las nuevas claves
        with open(output_file_path, "wb") as f:
            pickle.dump(
                {
                    "X_src": dataset["X_src"],
                    "X_tgt": dataset["X_tgt"],
                    "Y": dataset["Y"],
                    "ids": dataset["ids"],
                },
                f,
            )

        print(f"Dataset guardado en: {output_file_path} (Tamaño {len(dataset['X_src'])})")


def feature_expansion(X):
    """
    Expande las features añadiendo log(x) y log(1-x).
    Se asume que X contiene valores en el rango (0, 1).
    """
    X = np.array(X, dtype=np.float32)
    
    # Evitar log(0) o log(1)
    eps = 1e-6
    X_clamped = np.clip(X, eps, 1 - eps)
    
    # Calcular representaciones logarítmicas
    X_log = np.log(X_clamped)
    X_log_inv = np.log(1 - X_clamped)

    # Concatenar: [x, log(x), log(1-x)]
    X_expanded = np.concatenate([X_clamped, X_log, X_log_inv], axis=-1)

    return X_expanded

def normalize_input(X):
    # Escalar con StandardScaler
    # X shape: [num_ejemplos, num_acciones, 4]
    X = np.array(X, dtype=np.float32)

    # Aplano a 2D
    X_flat = X.reshape(-1, X.shape[-1])  # [num_ejemplos*num_acciones, 4]

    # Fit/transform
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # Vuelvo a la forma original
    X = X_scaled.reshape(-1, X.shape[1], X.shape[2])
    return X
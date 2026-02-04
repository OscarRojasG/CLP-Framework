import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
from data_generation import load_data
from settings import DATASETS_FOLDER


class Dataset(TensorDataset):
    def __init__(self, filepath, *tensors):
        super().__init__(*tensors)
        self.filepath = filepath
        self.name = os.path.basename(filepath)


def generate_datasets(filename, cuts, max_size=None, seed=42):
    """Separa los datos en varios datasets según los cortes y guarda en archivos .data"""
    np.random.seed(seed)

    # Añadir un "1" al principio del arreglo cuts para asegurar que el primer dataset sea considerado
    cuts = [1] + cuts

    # Asegurarse de que la carpeta de salida exista
    os.makedirs(DATASETS_FOLDER, exist_ok=True)

    # --- Cargar los datos ---
    print(f"Procesando archivo: {filename}")
    block_features, action_blocks, action_features, placed_blocks, placed_features, Y = load_data(filename)

    # --- Crear diccionario de datasets ---
    datasets = {
        i: {"block_features": [], "action_blocks": [], "action_features": [],
            "placed_blocks": [], "placed_features": [], "Y": []}
        for i in range(1, len(cuts))
    }

    # --- Separar los datos según los cortes ---
    for i, y_vector in enumerate(Y):
        pos = int(np.argmax(y_vector))

        for j in range(len(cuts) - 1):
            if pos < cuts[j + 1]:
                datasets[j + 1]["block_features"].append(block_features[i])
                datasets[j + 1]["action_blocks"].append(action_blocks[i])
                datasets[j + 1]["action_features"].append(action_features[i])
                datasets[j + 1]["placed_blocks"].append(placed_blocks[i])
                datasets[j + 1]["placed_features"].append(placed_features[i])
                datasets[j + 1]["Y"].append(Y[i])
                break

    # --- Guardar cada dataset ---
    for i, dataset in datasets.items():
        start_cut = cuts[i - 1] + 1 if i > 1 else 1
        end_cut = cuts[i]

        # Truncar si se excede el tamaño máximo
        if max_size is not None and len(dataset["block_features"]) > max_size:
            n = len(dataset["block_features"])
            indices = np.random.choice(n, size=max_size, replace=False)

            for key in dataset:
                dataset[key] = [dataset[key][idx] for idx in indices]

        output_filename = f"{filename.split('.')[0]}_{start_cut}-{end_cut}.data"
        output_file_path = DATASETS_FOLDER / output_filename

        # Guardar dataset con las nuevas claves
        with open(output_file_path, "wb") as f:
            pickle.dump(
                {
                    "block_features": dataset["block_features"],
                    "action_blocks": dataset["action_blocks"],
                    "action_features": dataset["action_features"],
                    "placed_blocks": dataset["placed_blocks"],
                    "placed_features": dataset["placed_features"],
                    "Y": dataset["Y"],
                },
                f,
            )

        print(f"Dataset guardado en: {output_file_path} (Tamaño {len(dataset['block_features'])})")


def load_dataset(filepath):
    block_features, action_blocks, action_features, placed_blocks, placed_features, Y = load_data(filepath)
    block_features = torch.tensor(np.array(block_features), dtype=torch.float32)
    action_blocks = torch.tensor(np.array(action_blocks), dtype=torch.int32)
    action_features = torch.tensor(np.array(action_features), dtype=torch.float32)
    placed_blocks = torch.tensor(np.array(placed_blocks), dtype=torch.int32)
    placed_features = torch.tensor(np.array(placed_features), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.int32)

    dataset = Dataset(filepath, block_features, action_blocks, action_features, placed_blocks, placed_features, Y)
    print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset
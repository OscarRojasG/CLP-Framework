import os
import numpy as np
import torch
import gc
import h5py
from torch.utils.data import Dataset
from settings import DATASETS_FOLDER, DATA_FOLDER


class H5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self._open_file()
        self.name = os.path.basename(file_path)

    def _open_file(self):
        self.file = h5py.File(self.file_path, "r")
        self.block_features = self.file["block_features"]
        self.action_blocks = self.file["action_blocks"]
        self.action_features = self.file["action_features"]
        self.placed_blocks = self.file["placed_blocks"]
        self.placed_features = self.file["placed_features"]
        self.Y = self.file["Y"]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.block_features[idx]),
            torch.from_numpy(self.action_blocks[idx]),
            torch.from_numpy(self.action_features[idx]),
            torch.from_numpy(self.placed_blocks[idx]),
            torch.from_numpy(self.placed_features[idx]),
            torch.tensor(self.Y[idx])
        )


def generate_datasets(filenames, basename, cuts, max_size=None, seed=42):
    """
    Separa los datos en varios datasets según los cortes y guarda en archivos .data
    a partir de uno o múltiples archivos de entrada.
    """
    np.random.seed(seed)

    os.makedirs(DATASETS_FOLDER, exist_ok=True)

    datasets = {
        i: {
            "block_features": [],
            "action_blocks": [],
            "action_features": [],
            "placed_blocks": [],
            "placed_features": [],
            "Y": [],
        }
        for i in range(1, len(cuts))
    }

    # --- Procesar cada archivo ---
    for filename in filenames:
        print(f"Procesando archivo: {filename}")
        dataset_obj = load_data(filename)

        for i in range(len(dataset_obj)):
            y_vector = dataset_obj.Y[i]
            pos = int(np.argmax(y_vector))

            for j in range(len(cuts) - 1):
                if pos < cuts[j + 1]:
                    datasets[j + 1]["block_features"].append(dataset_obj.block_features[i])
                    datasets[j + 1]["action_blocks"].append(dataset_obj.action_blocks[i])
                    datasets[j + 1]["action_features"].append(dataset_obj.action_features[i])
                    datasets[j + 1]["placed_blocks"].append(dataset_obj.placed_blocks[i])
                    datasets[j + 1]["placed_features"].append(dataset_obj.placed_features[i])
                    datasets[j + 1]["Y"].append(dataset_obj.Y[i])
                    break

        # cerrar archivo HDF5 explícitamente
        dataset_obj.file.close()
        gc.collect()

    # --- Guardar cada dataset en HDF5 ---
    for i, dataset in datasets.items():
        start_cut = cuts[i - 1] + 1 if i > 1 else cuts[i - 1]
        end_cut = cuts[i]

        if max_size is not None and len(dataset["block_features"]) > max_size:
            n = len(dataset["block_features"])
            indices = np.random.choice(n, size=max_size, replace=False)
            for key in dataset:
                dataset[key] = [dataset[key][idx] for idx in indices]

        output_filename = f"{basename}_{start_cut}-{end_cut}.data"
        output_file_path = DATASETS_FOLDER / output_filename

        # Convertir listas a arrays antes de guardar
        block_features_np = np.array(dataset["block_features"], dtype=np.float32)
        action_blocks_np = np.array(dataset["action_blocks"], dtype=np.int32)
        action_features_np = np.array(dataset["action_features"], dtype=np.float32)
        placed_blocks_np = np.array(dataset["placed_blocks"], dtype=np.int32)
        placed_features_np = np.array(dataset["placed_features"], dtype=np.float32)
        Y_np = np.array(dataset["Y"], dtype=np.int32)

        # Guardar en HDF5
        with h5py.File(output_file_path, "w") as f:
            f.create_dataset("block_features", data=block_features_np)
            f.create_dataset("action_blocks", data=action_blocks_np)
            f.create_dataset("action_features", data=action_features_np)
            f.create_dataset("placed_blocks", data=placed_blocks_np)
            f.create_dataset("placed_features", data=placed_features_np)
            f.create_dataset("Y", data=Y_np)

        print(
            f"Dataset guardado en: {output_file_path} "
            f"(Tamaño {len(block_features_np)})"
        )


def load_dataset(filepath):
    dataset = H5Dataset(DATASETS_FOLDER / filepath)
    print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset

def load_data(filepath):
    return H5Dataset(DATA_FOLDER / filepath)
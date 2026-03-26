import os
import numpy as np
import random
import torch
import gc
import h5py
from collections import defaultdict
from torch.utils.data import Dataset
from settings import DATASETS_FOLDER, DATA_FOLDER


class H5Dataset(Dataset):
    def __init__(self, file_path, lazy):
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        self.file = None
        
        with h5py.File(self.file_path, "r") as f:
            self.dataset_len = len(f["Y"])
        
        if not lazy:
            self._open_file()

    def _open_file(self):
        self.file = h5py.File(self.file_path, "r")
        self.block_features = self.file["block_features"]
        self.action_blocks = self.file["action_blocks"]
        self.action_features = self.file["action_features"]
        self.placed_blocks = self.file["placed_blocks"]
        self.placed_features = self.file["placed_features"]
        self.space_features = self.file["space_features"]
        self.Y = self.file["Y"]
        
    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.file is None:
            self._open_file()
            
        return (
            torch.from_numpy(self.block_features[idx]),
            torch.from_numpy(self.action_blocks[idx]),
            torch.from_numpy(self.action_features[idx]),
            torch.from_numpy(self.placed_blocks[idx]),
            torch.from_numpy(self.placed_features[idx]),
            torch.from_numpy(self.space_features[idx]),
            torch.tensor(self.Y[idx])
        )

def generate_datasets_(filename, start_cut, end_cut):
    """Retorna los índices dentro de un archivo que caen en el rango [start, end]."""
    dataset_obj = load_data(filename)
    indices_in_range = []
    
    # Solo recorremos Y para filtrar, evitando cargar todo a RAM
    for i in range(len(dataset_obj.Y)):
        pos = int(np.argmax(dataset_obj.Y[i]))
        if start_cut <= pos <= end_cut:
            indices_in_range.append(i)
            
    dataset_obj.close()
    return indices_in_range

def generate_datasets(filenames, basename, cuts, max_size=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(DATASETS_FOLDER, exist_ok=True)
    
    file_to_index = {fname: i for i, fname in enumerate(filenames)}

    # Iterar por cada rango de corte
    for j in range(len(cuts) - 1):
        start_cut = cuts[j] if j == 0 else cuts[j] + 1
        end_cut = cuts[j + 1]
        print(f"Procesando corte: [{start_cut}, {end_cut}]")

        # Recolectar todos los índices candidatos
        all_candidate_indices = [] # Lista de tuplas (filename, index)
        
        for filename in filenames:
            matched_indices = generate_datasets_(filename, start_cut, end_cut)
            for idx in matched_indices:
                all_candidate_indices.append((file_to_index[filename], idx))

        # Selección aleatoria
        if max_size and len(all_candidate_indices) > max_size:
            selected_indices = random.sample(all_candidate_indices, max_size)
        else:
            selected_indices = all_candidate_indices

        if not selected_indices:
            continue

        # Agrupamos por archivo para abrir cada archivo una sola vez en este corte
        indices_by_file = defaultdict(list)
        for f_id, idx in selected_indices:
            indices_by_file[f_id].append(idx)

        current_dataset = {k: [] for k in ["block_features", "action_blocks", "action_features", 
                                          "placed_blocks", "placed_features", "space_features", "Y"]}

        for f_id, indices in indices_by_file.items():
            filename = filenames[f_id]
            dataset_obj = load_data(filename)
            # Ordenar índices suele mejorar la velocidad de lectura en disco
            for idx in sorted(indices):
                current_dataset["block_features"].append(dataset_obj.block_features[idx])
                current_dataset["action_blocks"].append(dataset_obj.action_blocks[idx])
                current_dataset["action_features"].append(dataset_obj.action_features[idx])
                current_dataset["placed_blocks"].append(dataset_obj.placed_blocks[idx])
                current_dataset["placed_features"].append(dataset_obj.placed_features[idx])
                current_dataset["space_features"].append(dataset_obj.space_features[idx])
                current_dataset["Y"].append(dataset_obj.Y[idx])
            dataset_obj.close()

        # Guardar y liberar memoria
        save_to_h5(current_dataset, basename, start_cut, end_cut)
        
        del current_dataset
        gc.collect()

def save_to_h5(data_dict, basename, start, end):
    output_filename = f"{basename}_{start}-{end}.data"
    output_path = DATASETS_FOLDER / output_filename
    
    with h5py.File(output_path, "w") as f:
        for key, value in data_dict.items():
            dtype = np.int32 if key in ["Y", "action_blocks", "placed_blocks"] else np.float32
            f.create_dataset(key, data=np.array(value, dtype=dtype))
    
    print(f"Dataset guardado en: {output_path} (Tamaño {len(value)})")

def load_dataset(filepath):
    dataset = H5Dataset(DATASETS_FOLDER / filepath, lazy=True)
    print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset

def load_data(filepath):
    return H5Dataset(DATA_FOLDER / filepath, lazy=False)
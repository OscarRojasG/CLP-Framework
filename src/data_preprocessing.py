import os
import numpy as np
import random
import torch
import gc
import h5py
from collections import defaultdict
from torch.utils.data import Dataset
from settings import DATASETS_FOLDER, DATA_FOLDER
import settings
from misc.labels import LabelType


class H5Dataset(Dataset):
    def __init__(self, file_path, lazy):
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        self.file = None
        
        with h5py.File(self.file_path, "r") as f:
            self.dataset_len = len(f["Y"])
            self.label_type = f["Y"].attrs["label_type"] if "label_type" in f["Y"].attrs else LabelType.BEST_ACTION
        
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
    
class DatasetWithBias(H5Dataset):
    def __init__(self, file_path, lazy=False):
        super().__init__(file_path, lazy)

    def __getitem__(self, idx):
        # 1. Obtenemos los datos base de la clase padre
        data = super().__getitem__(idx)
        (block_features, action_blocks, action_features, 
         placed_blocks, placed_features, space_features, y) = data

        # 2. Extraemos los índices de bloques para las acciones
        # action_blocks tiene forma [Na]. Usamos clamp para evitar errores con el padding (-1)
        valid_indices = action_blocks.clamp(min=0).long()

        # 3. Extraemos componentes para la fórmula
        # block_features[idx][3] es x, [4] es y, etc.
        # Asumiendo block_features: [N_total_bloques, Dim_features]
        # Asumiendo action_features: [Na, Dim_action_features]
        
        vol = block_features[valid_indices, 3]
        n_frac = block_features[valid_indices, 4]
        
        # action_features[i][0] y [1]
        loss = action_features[:, 0]
        cs = action_features[:, 1]

        # 4. Aplicamos la fórmula de la heurística
        # h = x * y^γ * (1 - feat_2)^β * feat_3^α
        alpha=4.0
        beta=1.0
        gamma=0.2
        bias = (vol * (n_frac ** gamma) * ((1 - loss) ** beta) * (cs ** alpha))

        # 5. Manejo de Padding: Si action_blocks era -1, el bias debe ser -inf o 0 
        # para que no influya. log_softmax se encargará del resto en el modelo.
        mask = (action_blocks == -1)
        bias[mask] = -1 # O un valor muy bajo

        # Retornamos todo el pack original + el nuevo bias
        return (*data[:-1], bias, y)

def generate_datasets_(filename, start_cut, end_cut):
    """Retorna los índices dentro de un archivo que caen en el rango [start, end]."""
    dataset_obj = load_data(filename)
    indices_in_range = []
    
    # Solo recorremos Y para filtrar, evitando cargar todo a RAM
    for i in range(len(dataset_obj.Y)):
        pos = int(np.argmax(dataset_obj.Y[i]))
        if start_cut - 1 <= pos <= end_cut - 1:
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
        save_to_h5(current_dataset, basename, start_cut, end_cut, dataset_obj.label_type)
        
        del current_dataset
        gc.collect()

def save_to_h5(data_dict, basename, start, end, label_type):
    output_filename = f"{basename}_{start}-{end}.data"
    output_path = DATASETS_FOLDER / output_filename
    
    with h5py.File(output_path, "w") as f:
        for key, value in data_dict.items():
            dtype = np.int32 if key in ["action_blocks", "placed_blocks"] else np.float32
            f.create_dataset(key, data=np.array(value, dtype=dtype))
        f["Y"].attrs["label_type"] = label_type
    
    print(f"Dataset guardado en: {output_path} (Tamaño {len(value)})")

def load_dataset(filepath, bias=False):
    if bias:
        dataset = DatasetWithBias(settings.DATASETS_FOLDER / filepath, lazy=True)
    else:
        dataset = H5Dataset(settings.DATASETS_FOLDER / filepath, lazy=True)

    print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset

def load_data(filepath, lazy=False):
    return H5Dataset(DATA_FOLDER / filepath, lazy=lazy)
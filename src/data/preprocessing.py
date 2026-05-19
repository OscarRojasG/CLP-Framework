import os
import numpy as np
import random
import torch
import gc
import h5py
from collections import defaultdict
from torch.utils.data import Dataset
from settings import DATASETS_FOLDER, DATA_FOLDER


'''
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
'''
    

class H5Dataset(Dataset):
    def __init__(self, filepath, max_size=None):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.file = None

        with h5py.File(self.filepath, "r") as f:
            self.input_keys = list(f['input'].attrs['key_order'])
            self.output_keys = list(f['output'].attrs['key_order'])
            
            total_len = len(f['input'][self.input_keys[0]])
            self.dataset_len = total_len if max_size is None else min(total_len, max_size)

    def _open_file(self):
        self.file = h5py.File(self.filepath, "r")
        self.input_datasets = {k: self.file[f'input/{k}'] for k in self.input_keys}
        self.output_datasets = {k: self.file[f'output/{k}'] for k in self.output_keys}
        
    def _to_tensor(self, val):
        """Helper para convertir datos a tensores de forma eficiente"""
        if isinstance(val, np.ndarray):
            return torch.from_numpy(val)
        return torch.tensor(val)

    def __getitem__(self, idx):
        if self.file is None: 
            self._open_file()
            
        inputs = [self._to_tensor(self.input_datasets[k][idx]) for k in self.input_keys]
        outputs = [self._to_tensor(self.output_datasets[k][idx]) for k in self.output_keys]
        return tuple(inputs), tuple(outputs)
    
    def __len__(self):
        return self.dataset_len

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file'] = None
        state['input_datasets'] = None
        state['output_datasets'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.file = None


def load_data(filename):
    filepath = DATA_FOLDER / filename
    with h5py.File(filepath, "r") as f:
        # Extraemos keys de los grupos
        input_keys = list(f['input'].attrs['key_order'])
        output_keys = list(f['output'].attrs['key_order'])
        
        data = {}
        # Cargamos los inputs
        for k in input_keys:
            data[f'input/{k}'] = f['input'][k][:]
            
        # Cargamos los outputs
        for k in output_keys:
            data[f'output/{k}'] = f['output'][k][:]

        data['ranks'] = f['ranks'][:]

        # Guardamos metadatos de orden para la reconstrucción
        data['_input_order'] = input_keys
        data['_output_order'] = output_keys

        return data
    

def split_data(filename, start_cut, end_cut):
    """Retorna los índices dentro de un archivo que caen en el rango [start, end] basados en 'ranks'."""
    data = load_data(filename)
    
    # data['ranks'] es un array 1D. Evaluamos la condición de rango directamente
    # usando lógica vectorial de NumPy (mucho más rápido que un bucle for).
    ranks = data['ranks']
    condition = (start_cut - 1 <= ranks) & (ranks <= end_cut - 1)
    
    # Extraemos las posiciones (índices) donde la condición es True
    indices_in_range = np.where(condition)[0].tolist()
    
    return indices_in_range


def generate_datasets(filenames, basename, cuts, max_size=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(DATASETS_FOLDER, exist_ok=True)
    
    file_to_index = {fname: i for i, fname in enumerate(filenames)}

    for j in range(len(cuts) - 1):
        start_cut = cuts[j] if j == 0 else cuts[j] + 1
        end_cut = cuts[j + 1]
        print(f"Procesando corte: [{start_cut}, {end_cut}]")

        all_candidate_indices = []
        for filename in filenames:
            matched_indices = split_data(filename, start_cut, end_cut)
            for idx in matched_indices:
                all_candidate_indices.append((file_to_index[filename], idx))

        if max_size and len(all_candidate_indices) > max_size:
            selected_indices = random.sample(all_candidate_indices, max_size)
        else:
            selected_indices = all_candidate_indices

        if not selected_indices:
            continue

        indices_by_file = defaultdict(list)
        for f_id, idx in selected_indices:
            indices_by_file[f_id].append(idx)

        # Aquí acumularemos los bloques completos de arrays de cada archivo
        chunks_dataset = defaultdict(list)
        input_order = None
        output_order = None

        for f_id, indices in indices_by_file.items():
            filename = filenames[f_id]
            data = load_data(filename)
            
            if input_order is None:
                input_order = data['_input_order']
                output_order = data['_output_order']
            
            data_keys = [f'input/{k}' for k in data['_input_order']] + \
                        [f'output/{k}' for k in data['_output_order']] + \
                        ['ranks']
            
            # Aprovechamos que YA son arrays de NumPy:
            # Filtramos todas las filas deseadas de un solo golpe usando indexación avanzada
            sorted_indices = sorted(indices)
            for key in data_keys:
                chunks_dataset[key].append(data[key][sorted_indices])

        # Concatenamos los sub-arrays de cada archivo en un único array final por clave
        current_dataset = {}
        for key, arrays_list in chunks_dataset.items():
            current_dataset[key] = np.concatenate(arrays_list, axis=0)

        current_dataset['_input_order'] = input_order
        current_dataset['_output_order'] = output_order

        save_to_h5(current_dataset, basename, start_cut, end_cut)


def save_to_h5(data_dict, basename, start, end):
    output_filename = f"{basename}_{start}-{end}.data"
    output_path = DATASETS_FOLDER / output_filename
    
    input_order = data_dict['_input_order']
    output_order = data_dict['_output_order']
    
    with h5py.File(output_path, "w") as f:
        f.create_group('input')
        f.create_group('output')
        
        f['input'].attrs['key_order'] = input_order
        f['output'].attrs['key_order'] = output_order
        
        sample_length = 0
        for key, arr in data_dict.items():
            if key in ['_input_order', '_output_order']:
                continue
                
            # Pasamos el array de NumPy directamente sin conversiones redundantes
            f.create_dataset(key, data=arr)
            sample_length = len(arr)
            
    print(f"Dataset guardado en: {output_path} (Tamaño {sample_length})")


def load_dataset(filepath):
    dataset = H5Dataset(DATASETS_FOLDER / filepath)
    print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset
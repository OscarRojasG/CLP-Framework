import subprocess
import numpy as np

class Process():
    def __init__(self, process: subprocess.Popen):
        self.process = process

    def print_blocks(self):
        # Usamos b"" para enviar bytes directamente
        self.process.stdin.write(b"-B\n")
        self.process.stdin.flush()
        
    def read_exactly(self, n):
        """Lee exactamente n bytes del stdout."""
        data = b''
        while len(data) < n:
            packet = self.process.stdout.read(n - len(data))
            if not packet:
                raise EOFError("El proceso C++ se cerró inesperadamente")
            data += packet
        return data

    def read_blocks(self):
        # Leer header (4 bytes)
        header = self.read_exactly(4)
        num_blocks = np.frombuffer(header, dtype=np.uint32)[0]

        floats_per_block = 9
        bytes_to_read = num_blocks * floats_per_block * 4
        
        # Usar la nueva función para asegurar que leemos todo
        raw_data = self.read_exactly(bytes_to_read)
        
        # 3. Convertir a matriz de Numpy de forma instantánea
        all_data = np.frombuffer(raw_data, dtype=np.float32).reshape(num_blocks, floats_per_block)

        # Separar ID de métricas
        block_ids = all_data[:, 0].astype(int)
        metrics = all_data[:, 1:]

        # Reconstruir diccionarios (esto sigue siendo O(N), pero los datos ya están en memoria)
        id_to_index = {int(bid): i for i, bid in enumerate(block_ids)}
        index_to_id = {i: int(bid) for i, bid in enumerate(block_ids)}

        return metrics, id_to_index, index_to_id
    
    def print_placed_blocks(self):
        self.process.stdin.write(b"-P\n")
        self.process.stdin.flush()

    def read_placed_blocks(self, padding):
        # 1. Leer cuántos bloques vienen
        header = self.read_exactly(4)
        num_placed = np.frombuffer(header, dtype=np.uint32)[0]

        if num_placed == 0:
            # Manejar caso vacío con padding
            placed_features = np.full((padding, 4), -1.0, dtype=float)
            placed_blocks = np.full(padding, -1, dtype=int)
            return placed_blocks, placed_features

        # Cada bloque tiene 5 floats (id, bx, by, bz, contact)
        floats_per_row = 5
        raw_data = self.read_exactly(num_placed * floats_per_row * 4)
        
        # 2. Convertir a matriz
        all_data = np.frombuffer(raw_data, dtype=np.float32).reshape(num_placed, floats_per_row)

        # 3. Procesar IDs y Features
        # Convertimos IDs de C++ a índices locales de Python
        current_ids = all_data[:, 0].astype(int)
        placed_blocks = np.array([self.id_to_index[bid] for bid in current_ids])
        
        # Tomamos bx, by, bz, contact
        placed_features = all_data[:, 1:]

        # 4. Aplicar Padding
        n = placed_features.shape[0]
        pad_len = max(0, padding - n)

        if pad_len > 0:
            placed_features = np.pad(placed_features, ((0, pad_len), (0, 0)), 
                                    mode='constant', constant_values=-1.0)
            placed_blocks = np.pad(placed_blocks, (0, pad_len), 
                                mode='constant', constant_values=-1)

        return placed_blocks, placed_features
    
    def print_space(self):
        self.process.stdin.write(b"-S\n")
        self.process.stdin.flush()
        
    def read_space(self):
        # Son 6 floats * 4 bytes cada uno = 24 bytes
        num_features = 6
        bytes_to_read = num_features * 4
        
        raw_data = self.read_exactly(bytes_to_read)
        
        # Convertimos directamente a array de numpy
        return np.frombuffer(raw_data, dtype=np.float32)
    
    def print_actions(self):
        self.process.stdin.write(b"-A\n")
        self.process.stdin.flush()

    def read_actions(self, padding):
        # 1. Leer cuántas acciones vienen (4 bytes)
        header = self.read_exactly(4)
        num_actions = np.frombuffer(header, dtype=np.uint32)[0]
        
        # Definir cuántos floats tiene cada fila (ID + métricas)
        floats_per_row = 3
        num_features = floats_per_row - 1

        # Caso: No vienen acciones o el buffer está vacío
        if num_actions == 0:
            action_blocks = np.full(padding, -1, dtype=int)
            action_features = np.full((padding, num_features), -1.0, dtype=np.float32)
            return action_blocks, action_features

        # 2. Leer datos y convertir a matriz
        bytes_to_read = num_actions * floats_per_row * 4
        raw_data = self.read_exactly(bytes_to_read)
        all_data = np.frombuffer(raw_data, dtype=np.float32).reshape(num_actions, floats_per_row)

        # 3. Separar IDs y convertirlos a índices locales
        action_ids = all_data[:, 0].astype(int)
        # Nota: Asegúrate de que todos los aid existan en id_to_index
        action_blocks = np.array([self.id_to_index[aid] for aid in action_ids])
        action_features = all_data[:, 1:]

        # 4. Aplicar Padding (Lógica idéntica a read_placed_blocks)
        n = action_features.shape[0]
        pad_len = max(0, padding - n)

        if pad_len > 0:
            # Rellenar IDs con -1
            action_blocks = np.pad(action_blocks, (0, pad_len), 
                                mode='constant', constant_values=-1)
            # Rellenar Features con -1.0
            # La tupla ((0, pad_len), (0, 0)) indica: pad al final de las filas, nada en columnas
            action_features = np.pad(action_features, ((0, pad_len), (0, 0)), 
                                    mode='constant', constant_values=-1.0)

        return action_blocks, action_features
    
    def print_volume_ratio(self):
        self.process.stdin.write(b"-V\n")
        self.process.stdin.flush()

    def read_volume_ratio(self):
        # Un float son 4 bytes
        raw_data = self.read_exactly(4)
        
        # Convertimos y obtenemos el primer (y único) elemento
        value = np.frombuffer(raw_data, dtype=np.float32)[0]
        
        return float(value)

    def close(self):
        self.process.stdin.write(b"-Q\n")
        self.process.stdin.flush()
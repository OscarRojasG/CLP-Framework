import subprocess
import numpy as np

class Process():
    def __init__(self, process: subprocess.Popen):
        self.process = process

    def print_blocks(self):
        cmd = f"-B\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()

    def read_blocks(self):
        # Diccionario de índices
        id_to_index = {}
        index_to_id = {}

        # Leer hasta que no haya más salida (bloquea si el proceso no termina)
        output = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line == "END":
                break
            parts = line.split()
            if len(parts) > 1:
                # Ignora el primer elemento (block_id)
                metrics = [float(x) for x in parts[1:]]
                id_to_index[int(parts[0])] = len(output)
                index_to_id[len(output)] = int(parts[0])
                output.append(metrics)
        return np.array(output), id_to_index, index_to_id
    
    def print_placed_blocks(self):
        cmd = "-P\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()

    def read_placed_blocks(self, padding):
        placed_blocks = []
        placed_features = []

        # Leer todas las líneas hasta que no haya más salida
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line == "END":
                break

            parts = line.split()

            block_id = int(parts[0])
            features = list(map(float, parts[1:]))

            placed_blocks.append(self.id_to_index[block_id])
            placed_features.append(features)

        placed_features = np.array(placed_features, dtype=float).reshape(-1, 4)
        placed_blocks = np.array(placed_blocks, dtype=int)

        n = placed_features.shape[0]
        pad_len = max(0, padding - n)

        if pad_len > 0:
            placed_features = np.pad(placed_features, pad_width=((0, pad_len), (0, 0)), mode='constant', constant_values=-1)
            placed_blocks = np.pad(placed_blocks, pad_width=(0, pad_len), mode='constant', constant_values=-1)

        return placed_blocks, placed_features
    
    def print_space(self):
        cmd = "-S\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()
        
    def read_space(self):
        line = self.process.stdout.readline().strip()
        self.process.stdout.readline() # END
        space_features = [float(x) for x in line.split()]
        return np.array(space_features, dtype=float)
    
    def print_actions(self):
        cmd = "-A\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()

    def read_actions(self):
        action_blocks = []
        action_features = []

        # Leer la salida línea por línea
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line == "END":
                break

            parts = line.split()

            block_id = int(parts[0])
            # [2:] Ignoramos block_id + eval (VCS)
            values = [float(x) for x in parts[2:]]

            action_blocks.append(self.id_to_index[block_id])
            action_features.append(values)

        action_blocks = np.array(action_blocks, dtype=int)
        action_features = np.array(action_features, dtype=float)

        return action_blocks, action_features
    
    def print_volume_ratio(self):
        cmd = "-V\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()

    def read_volume_ratio(self):
        line = self.process.stdout.readline().strip()
        self.process.stdout.readline() # END
        value = float(line)
        return value

    def close(self):
        self.process.stdin.write("-Q\n")
        self.process.stdin.flush()
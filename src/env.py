import subprocess
import numpy as np
from settings import BSG_ENV_PATH, INSTANCE_FOLDER

class State():
    def __init__(self, process: subprocess.Popen):
        self.process = process  # Referencia al proceso persistente
        self.blocks, self.id_to_index, self.index_to_id = self.process_get_blocks()
        self.process_update()

    def process_get_blocks(self):
        cmd = f"-B\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()

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

    def process_get_placed_blocks(self, padding=64):
        cmd = "-P\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()

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

        placed_features = np.array(placed_features, dtype=float).reshape(-1, 3)
        placed_blocks = np.array(placed_blocks, dtype=int)

        n = placed_features.shape[0]
        pad_len = max(0, padding - n)

        if pad_len > 0:
            placed_features = np.pad(placed_features, pad_width=((0, pad_len), (0, 0)), mode='constant', constant_values=-1)
            placed_blocks = np.pad(placed_blocks, pad_width=(0, pad_len), mode='constant', constant_values=-1)

        return placed_blocks, placed_features
    
    def process_get_actions(self):
        cmd = "-A\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()

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
    
    def process_get_volume_ratio(self):
        cmd = "-V\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()

        # Leer una sola línea y convertirla a float
        line = self.process.stdout.readline().strip()
        value = float(line)

        return value
    
    def process_update(self):
        self.volume_ratio = self.process_get_volume_ratio()
        self.action_blocks, self.action_features = self.process_get_actions()
        if len(self.action_blocks) == 0: return

        self.placed_blocks, self.placed_features = self.process_get_placed_blocks()
    
    def get_block_features(self):
        return self.blocks
    
    def get_action_blocks(self):
        return self.action_blocks
    
    def get_action_features(self):
        return self.action_features
    
    def get_placed_blocks(self):
        return self.placed_blocks
    
    def get_placed_features(self):
        return self.placed_features
    
    def get_volume_ratio(self):
        return self.volume_ratio

    def close(self):
        self.process.stdin.write("-Q\n")
        self.process.stdin.flush()
    
class Action:
    def __init__(self, block_id, action_vec):
        self.block_id = block_id
        self.action_vec = action_vec

class Environment:
    @staticmethod
    def initial_state(instance_file, instance_number, w: int) -> State:
        """
        Inicia el proceso persistente de BSG_ENV y configura el estado inicial.
        """
        # Crear proceso persistente
        process = subprocess.Popen(
            [
                BSG_ENV_PATH,
                INSTANCE_FOLDER / instance_file,
                "-i", str(instance_number),
                "-w", str(w)
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        process.stdout.readline() # Leer línea inicial
        return State(process)

    @staticmethod
    def state_transition(state: State, action_block: int):
        """
        Envía una acción al proceso para realizar la transición de estado
        y devuelve el valor flotante 'reward' resultante.
        """
        process = state.process
        block_id = state.index_to_id[action_block]

        cmd = f"-T {block_id}\n"
        process.stdin.write(cmd)
        process.stdin.flush()

        # Leer una línea del stdout con el reward)
        line = process.stdout.readline().strip()
        reward = float(line)

        # Actualizar variables de estado
        state.process_update()
        return reward
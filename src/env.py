import subprocess
import numpy as np

class State():
    def __init__(self, process: subprocess.Popen):
        self.process = process  # Referencia al proceso persistente
        self.blocks, self.block_index_dict = self.process_get_blocks()
        self.process_update()

    def process_get_blocks(self):
        cmd = f"-B\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()

        # Diccionario de índices
        block_index_dict = {}

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
                block_index_dict[int(parts[0])] = len(output)
                output.append(metrics)
        return np.array(output), block_index_dict

    def process_get_placed_blocks(self, block_index_dict, padding=64):
        cmd = "-P\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()

        placed = []
        placed_coords = []

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
            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

            placed.append(block_index_dict[block_id])
            placed_coords.append(coords)

        # Aplicar padding si hay menos de `padding` bloques
        if len(placed) < padding:
            placed.extend([-1] * (padding - len(placed)))
            placed_coords.extend([[0.0, 0.0, 0.0]] * (padding - len(placed_coords)))

        return np.array(placed), np.array(placed_coords)
    
    def process_get_actions(self):
        cmd = "-A\n"
        self.process.stdin.write(cmd)
        self.process.stdin.flush()

        block_ids = []
        metrics = []

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

            block_ids.append(block_id)
            metrics.append(values)

        # Convertir la lista de métricas a un array de NumPy
        metrics_array = np.array(metrics, dtype=float)

        return metrics_array, block_ids
    
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
        self.actions, self.action_blocks_id = self.process_get_actions()
        if len(self.actions) == 0: return

        self.placed, self.coords_placed = self.process_get_placed_blocks(self.block_index_dict)
    
    def get_blocks(self):
        return self.blocks
    
    def get_placed(self):
        return self.placed
    
    def get_coords(self):
        return self.coords_placed
    
    def get_actions(self, add_block_index=False):
        actions = []
        for i in range(len(self.actions)):
            block_id = self.action_blocks_id[i]
            action = self.actions[i]
            if add_block_index:
                action = np.insert(action, 0, self.block_index_dict[block_id], axis=0)
            actions.append(Action(block_id, action))
        return actions
    
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
                "./BSG_ENV",
                f"instances/{instance_file}",
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
    def get_valid_actions(state: State, add_block_index: bool) -> list[Action]:
        return state.get_actions(add_block_index)

    @staticmethod
    def state_transition(state: State, action: Action):
        """
        Envía una acción al proceso para realizar la transición de estado
        y devuelve el valor flotante 'reward' resultante.
        """
        process = state.process
        cmd = f"-T {action.block_id}\n"
        process.stdin.write(cmd)
        process.stdin.flush()

        # Leer una línea del stdout con el reward)
        line = process.stdout.readline().strip()
        reward = float(line)

        # Actualizar variables de estado
        state.process_update()
        return reward
import subprocess
import numpy as np
from envs.process import Process
from settings import BSG_ENV_PATH, INSTANCE_FOLDER

class State(Process):
    def __init__(self, process: subprocess.Popen):
        super().__init__(process)
        self.blocks, self.id_to_index, self.index_to_id = self.process_get_blocks()
        self.update()
    
    def update(self):
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
        state.update()
        return reward
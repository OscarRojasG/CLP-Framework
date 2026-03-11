import subprocess
import numpy as np
import torch
from envs.process import Process
from settings import BSG_ENV_PATH, INSTANCE_FOLDER

class BSM(Process):
    def __init__(self, process: subprocess.Popen, w: int):
        super().__init__(process)
        self.num_states = 1
        self.w = w

        self.print_blocks()
        self.block_features, self.id_to_index, self.index_to_id = self.read_blocks()
        self.update()

    def read_placed_blocks(self, padding=64):
        placed_blocks_batch = []
        placed_features_batch = []

        for _ in range(self.num_states):
            placed_blocks, placed_features = super().read_placed_blocks(padding)
            placed_blocks_batch.append(placed_blocks)
            placed_features_batch.append(placed_features)

        return torch.from_numpy(np.array(placed_blocks_batch)).int(), torch.from_numpy(np.array(placed_features_batch)).float()
    
    def read_actions(self):
        action_blocks_batch = []
        action_features_batch = []

        for _ in range(self.num_states):
            action_blocks, action_features = super().read_actions(padding=self.w*self.w)
            action_blocks_batch.append(action_blocks)
            action_features_batch.append(action_features)

        return torch.from_numpy(np.array(action_blocks_batch)).int(), torch.from_numpy(np.array(action_features_batch)).float()
    
    def read_spaces(self):
        spaces_batch = []

        for _ in range(self.num_states):
            space_features = super().read_space()
            spaces_batch.append(space_features)

        return torch.from_numpy(np.array(spaces_batch)).float()
    
    def update(self):
        if self.num_states == 0: return
        super().print_volume_ratio()
        self.volume_ratio = super().read_volume_ratio()
        super().print_actions()
        self.action_blocks_batch, self.action_features_batch = self.read_actions()
        super().print_placed_blocks()
        self.placed_blocks_batch, self.placed_features_batch = self.read_placed_blocks()
        super().print_space()
        self.space_features_batch = self.read_spaces()

    
class BSMGreedyProcess():
    def __init__(self, bsm: BSM):
        self.bsm = bsm
        self.index_to_id = bsm.index_to_id
        self.id_to_index = bsm.id_to_index
        self.finished = False
        self.update()
        
    def update(self):
        if self.finished: return
        
        # 1. Leer el byte de señal (0 = datos, 1 = fin)
        signal_raw = self.bsm.process.stdout.read(1)
        if not signal_raw: return
        
        signal = np.frombuffer(signal_raw, dtype=np.uint8)[0]
        
        if signal == 1:
            self.finished = True
            # Leer el num_states (4 bytes)
            num_states_raw = self.bsm.read_exactly(4)
            self.bsm.num_states = int(np.frombuffer(num_states_raw, dtype=np.uint32)[0])
            return
                
        # 2. Si signal es 0, leer los bloques de datos en el orden exacto
        num_states_raw = self.bsm.read_exactly(4)
        self.bsm.num_states = int(np.frombuffer(num_states_raw, dtype=np.uint32)[0])
        
        self.action_blocks_batch, self.action_features_batch = self.bsm.read_actions()
        self.placed_blocks_batch, self.placed_features_batch = self.bsm.read_placed_blocks()
        self.space_features_batch = self.bsm.read_spaces()

    def transition(self, action_block_list):
        action_ids = [str(self.index_to_id[idx]) for idx in action_block_list]
        
        # Enviar respuesta como una única línea
        response = " ".join(action_ids) + "\n"
        self.bsm.process.stdin.write(response.encode('utf-8'))
        self.bsm.process.stdin.flush()
        self.update()
        
        

class BSMEnvironment:
    @staticmethod
    def init(instance_file, instance_number, w: int) -> BSM:
        # Crear proceso persistente
        process = subprocess.Popen(
            [
                BSG_ENV_PATH,
                INSTANCE_FOLDER / instance_file,
                "-i", str(instance_number),
                "-w", str(w),
                "--bsm"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0
        )

        process.stdout.readline() # Leer línea inicial
        return BSM(process, w)

    @staticmethod
    def next(bsm: BSM, action_blocks_batch):
        process = bsm.process

        block_ids_batch = []
        for action_blocks in action_blocks_batch:
            # Convertimos los índices de la red neuronal/numpy a los IDs de C++
            block_ids = [bsm.index_to_id[int(action_block)] for action_block in action_blocks]
            block_ids_batch.append(block_ids)

        # Generamos el string de IDs
        str_block_ids = ";".join(
            ",".join(map(str, block_ids)) for block_ids in block_ids_batch
        )

        # IMPORTANTE: Construimos el comando y lo codificamos a UTF-8/ASCII
        cmd = f"-T {str_block_ids}\n".encode('utf-8')
        
        process.stdin.write(cmd)
        process.stdin.flush()
        
        return BSMGreedyProcess(bsm)
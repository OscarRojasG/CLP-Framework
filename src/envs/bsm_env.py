import subprocess
import numpy as np
from envs.process import Process
from settings import BSG_ENV_PATH, INSTANCE_FOLDER

class BSM(Process):
    def __init__(self, process: subprocess.Popen):
        super().__init__(process)
        self.num_states = 1

        self.print_blocks()
        self.blocks, self.id_to_index, self.index_to_id = self.read_blocks()
        self.update()

    def read_placed_blocks(self, padding=64):
        placed_blocks_batch = []
        placed_features_batch = []
        super().print_placed_blocks()

        for _ in range(self.num_states):
            placed_blocks, placed_features = super().read_placed_blocks(padding)
            placed_blocks_batch.append(placed_blocks)
            placed_features_batch.append(placed_features)

        return placed_blocks_batch, placed_features_batch
    
    def read_actions(self):
        action_blocks_batch = []
        action_features_batch = []
        super().print_actions()

        for _ in range(self.num_states):
            action_blocks, action_features = super().read_actions()
            action_blocks_batch.append(action_blocks)
            action_features_batch.append(action_features)

        return action_blocks_batch, action_features_batch
    
    def read_spaces(self):
        spaces_batch = []
        super().print_space()

        for _ in range(self.num_states):
            space_features = super().read_space()
            spaces_batch.append(space_features)

        return spaces_batch
    
    def update(self):
        if self.num_states == 0: return

        super().print_volume_ratio()
        self.volume_ratio = super().read_volume_ratio()
        self.action_blocks_batch, self.action_features_batch = self.read_actions()
        self.placed_blocks_batch, self.placed_features_batch = self.read_placed_blocks()
        self.space_features_batch = self.read_spaces()
    
    def get_block_features(self):
        return self.blocks
    
    def get_action_blocks_batch(self):
        return self.action_blocks_batch
    
    def get_action_features_batch(self):
        return self.action_features_batch
    
    def get_placed_blocks_batch(self):
        return self.placed_blocks_batch
    
    def get_placed_features_batch(self):
        return self.placed_features_batch
    
    def get_space_features_batch(self):
        return self.space_features_batch
    
    def get_volume_ratio(self):
        return self.volume_ratio

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
            text=True,
            bufsize=1
        )

        process.stdout.readline() # Leer línea inicial
        return BSM(process)

    @staticmethod
    def next(bsm: BSM, action_blocks_batch):
        process = bsm.process

        block_ids_batch = []
        for action_blocks in action_blocks_batch:
            block_ids = [bsm.index_to_id[int(action_block)] for action_block in action_blocks]
            block_ids_batch.append(block_ids)

        str_block_ids = ";".join(
            ",".join(map(str, block_ids)) for block_ids in block_ids_batch
        )

        cmd = f"-T {str_block_ids}\n"
        process.stdin.write(cmd)
        process.stdin.flush()

        line = process.stdout.readline().strip()
        bsm.num_states = int(line)
        bsm.update()

        return bsm.num_states
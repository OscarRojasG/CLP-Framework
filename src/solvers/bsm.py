from envs.bsm_env import BSMEnvironment, BSM
from models.base.transformer import Transformer
import torch
import numpy as np

class BSMSolver():
    def __init__(self, model: Transformer):
        self.model = model
        self.env = BSMEnvironment()

    def solve(self, instance_file, instance_number, w: int) -> int:
        bsm = self.env.init(instance_file, instance_number, w)
        volume = self._solve(w, bsm)
        bsm.close()
        return volume

    def _solve(self, w: int, bsm: BSM) -> int:
        block_features = torch.tensor(np.array([bsm.get_block_features()]), dtype=torch.float32)
        memory = self.model.encode(block_features)

        while True:
            try:
                action_blocks_batch = bsm.get_action_blocks_batch()
                action_features_batch = bsm.get_action_features_batch()
                placed_blocks_batch = bsm.get_placed_blocks_batch()
                placed_features_batch = bsm.get_placed_features_batch()
            except Exception as e:
                print("Error obteniendo datos:", e)
                bsm.close()
                raise

            action_indexes_batch = []
            for i in range(len(action_blocks_batch)):
                action_blocks = torch.tensor(np.array([action_blocks_batch[i]]), dtype=torch.int32)
                action_features = torch.tensor(np.array([action_features_batch[i]]), dtype=torch.float32)
                placed_blocks = torch.tensor(np.array([placed_blocks_batch[i]]), dtype=torch.int32)
                placed_features = torch.tensor(np.array([placed_features_batch[i]]), dtype=torch.float32)

                # Predecir la mejor acción
                output = self.model.decode(memory, action_blocks, action_features, placed_blocks, placed_features)
                _, action_indexes = output.topk(min(w, len(output[0])), dim=1)
                action_indexes_batch.append(action_indexes[0])

            selected_action_blocks_batch = []
            for i in range(len(action_indexes_batch)):
                selected_action_blocks = []
                for j in range(len(action_indexes_batch[i])):
                    selected_action_blocks.append(action_blocks_batch[i][action_indexes_batch[i][j]].item())
                selected_action_blocks_batch.append(selected_action_blocks)

            # Aplicar la acción
            num_states = self.env.next(bsm, selected_action_blocks_batch)
            if num_states == 0: break
            
        return bsm.get_volume_ratio() * 100
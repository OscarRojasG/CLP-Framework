from envs.env import Environment
from models.base.transformer import Transformer
import torch
import numpy as np

class GreedyModelSolver():
    def __init__(self, model: Transformer):
        self.model = model

    def solve(self, instance_file, instance_number, w: int) -> int:
        env = Environment()
        state = env.initial_state(instance_file, instance_number, w)
        block_features = torch.tensor(np.array([state.get_block_features()]), dtype=torch.float32)
        memory = self.model.encode(block_features)

        while True:
            try:
                action_blocks = torch.tensor(np.array([state.get_action_blocks()]), dtype=torch.int32)
                action_features = torch.tensor(np.array([state.get_action_features()]), dtype=torch.float32)
                placed_blocks = torch.tensor(np.array([state.get_placed_blocks()]), dtype=torch.int32)
                placed_features = torch.tensor(np.array([state.get_placed_features()]), dtype=torch.float32)
                space_features = torch.tensor(np.array([state.get_space_features()]), dtype=torch.float32)
            except Exception as e:
                print("Error obteniendo datos:", e)
                state.close()
                raise

            if len(state.get_action_blocks()) == 0: break # Estado completado

            # Predecir la mejor acción
            output = self.model.decode(memory, action_blocks, action_features, placed_blocks, placed_features, space_features)
            action_block = int(action_blocks[0][output.argmax()])

            # Aplicar la acción
            env.state_transition(state, action_block)
            
        state.close()
        return state.get_volume_ratio() * 100
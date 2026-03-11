import torch
import numpy as np
from abc import ABC, abstractmethod
from envs.env import Environment
from models.base.transformer import Transformer

class BaseSolver(ABC):
    def __init__(self, model: Transformer):
        self.model = model

    @abstractmethod
    def select_action(self, logits: torch.Tensor, action_blocks: torch.Tensor) -> int:
        """
        Define la estrategia de selección: argmax, muestreo estocástico, etc.
        """
        pass

    def solve(self, instance_file, instance_number, w: int) -> float:
        env = Environment()
        state = env.initial_state(instance_file, instance_number, w)
        
        # Codificación inicial (fuera del loop para eficiencia si la arquitectura lo permite)
        block_features = torch.tensor(np.array([state.get_block_features()]), dtype=torch.float32)
        memory = self.model.encode(block_features)

        while True:
            current_actions = state.get_action_blocks()
            if len(current_actions) == 0:
                break 

            try:
                # Preparación de tensores
                action_blocks = torch.tensor(np.array([current_actions]), dtype=torch.int32)
                action_features = torch.tensor(np.array([state.get_action_features()]), dtype=torch.float32)
                placed_blocks = torch.tensor(np.array([state.get_placed_blocks()]), dtype=torch.int32)
                placed_features = torch.tensor(np.array([state.get_placed_features()]), dtype=torch.float32)
                space_features = torch.tensor(np.array([state.get_space_features()]), dtype=torch.float32)
                
                # Inferencia del modelo
                logits = self.model.decode(
                    memory, action_blocks, action_features, 
                    placed_blocks, placed_features, space_features
                )
                
                # Selección de acción delegada a la subclase
                try:
                    action_block = self.select_action(logits[0], action_blocks[0])
                except Exception as e:
                    print("Action blocks", action_blocks)
                    print("Action features", action_features)
                    print("Placed blocks", placed_blocks)
                    print("Placed features", placed_features)
                    print("Space features", space_features)
                    raise e
                #action_block = self.select_action(logits[0], action_blocks[0])
                
                # Transición
                env.state_transition(state, action_block)

            except Exception as e:
                state.close()
                raise e
        
        result = state.get_volume_ratio() * 100
        state.close()
        return result
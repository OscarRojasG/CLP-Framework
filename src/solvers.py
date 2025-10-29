from torch import nn
from .env import Environment, State
from abc import ABC, abstractmethod
import subprocess
from . import settings
from src.data_preprocessing import feature_expansion, normalize_input
import torch

class Solver(ABC):
    @abstractmethod
    def solve(self, instance_file, instance_number, w: int) -> int:
        pass

class ModelSolver(Solver):
    def __init__(self, model: nn.Module):
        self.model = model

    def solve(self, instance_file, instance_number, w: int) -> int:
        env = Environment
        state = env.initial_state(instance_file, instance_number, w)

        while True:
            try:
                valid_actions = env.get_valid_actions(state)
            except Exception as e:
                print("Error obteniendo acciones válidas:", e)
                state.close()
                raise

            if len(valid_actions) == 0: break # Estado completado

            # Convertir acciones a vectores
            action_vec = [[action.action_vec for action in valid_actions]]
            action_vec = feature_expansion(action_vec)
            action_vec = normalize_input(action_vec)
            action_vec = torch.tensor(action_vec, dtype=torch.float32)

            # Predecir la mejor acción
            action_idx = self.model(action_vec).argmax()
            seledted_action = valid_actions[action_idx]

            # Aplicar la acción
            state = env.state_transition(state, seledted_action)

        state.close()
        return state.get_volume_ratio()

class BSGSolver(Solver):
    def solve(self, instance_file, instance_number, w: int) -> int:
        # Ejecutar el proceso y capturar la salida
        proc = subprocess.run(
            ["./BSG_CLP", settings.instance_folder_path+instance_file, "-i", str(instance_number), "-w", str(w)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )

        line = proc.stdout.strip().splitlines()[-1]
        volume = float(line)
        return volume

class VCSSolver(BSGSolver):
    def solve(self, instance_file, instance_number):
        return super().solve(instance_file, instance_number, 1)
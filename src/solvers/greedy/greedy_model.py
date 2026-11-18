import torch
from solvers.solver import Solver
from settings import INSTANCE_FOLDER
import os
from data.objects import *
from solvers.env_solver import EnvSolver


class GreedyModelSolver(Solver, EnvSolver):
    def __init__(self, model, input_adapter, w, min_fr):
        Solver.__init__(self, "GreedyModel", min_fr)
        self.model = model
        self.w = w
        self.input_adapter = input_adapter

    def solve(self, instance_file, instance_number):
        from envs.clp_engine import GreedyModel

        instance_file = str(INSTANCE_FOLDER / instance_file)

        if not os.path.exists(instance_file):
            raise Exception(f'El archivo de instancia {instance_file} no existe.')

        device = next(self.model.parameters()).device
        env = GreedyModel(instance_file, instance_number, self.w,
                          self.input_adapter.max_blocks,
                          self.input_adapter.max_actions,
                          self.input_adapter.max_pblocks,
                          self.min_fr)

        with torch.inference_mode():
            enc_tensors = self.get_enc_inputs(env, device)
            memory = self.model.encode(*enc_tensors)

            while not env.is_finished():
                dec_tensors = self.get_dec_inputs(env, device)
                output = self.model.decode(*memory, *dec_tensors)
                best_index = output.argmax(dim=1).item()
                self.apply_transition(env, best_index, dec_tensors)

        vol = env.volume * 100
        time = env.final_time
        del env

        return vol, time

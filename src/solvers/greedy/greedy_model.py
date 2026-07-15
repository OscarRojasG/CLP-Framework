import torch
from solvers.solver import Solver
from settings import INSTANCE_FOLDER
import os
from data.objects import *
from solvers.env_solver import EnvSolver


class GreedyModelSolver(Solver, EnvSolver): 
    def __init__(self, model, input_adapter, w, min_fr, inference_mode=True):
        Solver.__init__(self, "GreedyModel", min_fr)
        EnvSolver.__init__(self, inference_mode)
        self.model = model
        self.w = w
        self.input_adapter = input_adapter
        
        # Importación condicional limpia
        if self.inference_mode:
            from envs.bsm_engine_inf import GreedyModel
            self.GreedyModel = GreedyModel
        else:
            from envs.bsm_engine_dev import GreedyModel
            self.GreedyModel = GreedyModel

    def solve(self, instance_file, instance_number):
        instance_file = str(INSTANCE_FOLDER / instance_file) 
        
        if not os.path.exists(instance_file):
            raise Exception(f'El archivo de instancia {instance_file} no existe.')
        
        device = next(self.model.parameters()).device

        # 1. Instanciación dependiente del modo
        if self.inference_mode:
            env = self.GreedyModel(instance_file, instance_number, self.w, 
                                   self.input_adapter.max_blocks,
                                   self.input_adapter.max_actions,
                                   self.input_adapter.max_pblocks,
                                   self.min_fr)
            block_data = None
        else:
            env = self.GreedyModel(instance_file, instance_number, self.w, self.min_fr)
            block_data = self.process_block_data(env.get_block_data())

        # 2. Context manager dinámico para máxima velocidad
        context_mgr = torch.inference_mode() if self.inference_mode else torch.no_grad()
        
        with context_mgr:
            # Extracción limpia del Encoder
            enc_tensors = self.get_enc_inputs(env, block_data, device)
            memory = self.model.encode(*enc_tensors)

            # Bucle inmaculado
            while not env.is_finished():
                # Extracción limpia del Decoder
                dec_tensors, action_data = self.get_dec_inputs(env, block_data, device)

                output = self.model.decode(*memory, *dec_tensors)
                best_index = output.argmax(dim=1).item()
                
                # Transición limpia
                self.apply_transition(env, best_index, action_data, dec_tensors)

        vol = env.volume * 100
        time = env.final_time
        del env
        
        return vol, time
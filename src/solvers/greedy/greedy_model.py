import torch
from bsm_engine import GreedyModel
from solvers.solver import Solver
from settings import INSTANCE_FOLDER
import os
from data.objects import *
from solvers.env_solver import EnvSolver


class GreedyModelSolver(Solver, EnvSolver): 
    def __init__(self, model, w, input_adapter, min_fr):
        Solver.__init__(self, "GreedyModel", min_fr)
        self.model = model
        self.w = w
        self.input_adapter = input_adapter

    def solve(self, instance_file, instance_number):
        instance_file = str(INSTANCE_FOLDER / instance_file) 
        
        if os.path.exists(instance_file) == False:
            raise Exception(f'El archivo de instancia {instance_file} no existe.')
        
        env = GreedyModel(instance_file, instance_number, self.w, self.min_fr)
        
        block_data = self.process_block_data(env.get_block_data())

        enc_data = self.input_adapter.enc_2_vec(block_data)
        enc_data = tuple(torch.from_numpy(data).unsqueeze(0) for data in enc_data)
        
        with torch.no_grad():
            enc_data = self.model.encode(*enc_data)

            while not env.is_finished():
                space_data = self.process_space_data(env.get_space_data())
                pblock_data = self.process_pblock_data(env.get_pblock_data())
                action_data = self.process_action_data(env.get_action_data())

                dec_data = self.input_adapter.dec_2_vec(block_data, space_data, pblock_data, action_data)
                dec_data = tuple(torch.from_numpy(data).unsqueeze(0) for data in dec_data)

                output = self.model.decode(*enc_data, *dec_data)

                best_index = output.argmax(dim=1).item()
                selected_block = action_data[best_index].block_id
                
                env.transition(selected_block)

        vol = env.volume * 100
        time = env.final_time
        del env
        return vol, time
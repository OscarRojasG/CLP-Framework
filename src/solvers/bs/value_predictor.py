from bsm_engine import ValuePredictor
from solvers.bs.bsm_solver import BSMSolver
from solvers.env_solver import EnvSolver
import torch
from data.objects import *
import numpy as np

class BSGValuePredictor(BSMSolver, EnvSolver):
    def __init__(self, action_model, value_model, action_input_adapter, value_input_adapter, w, min_fr):
        BSMSolver.__init__(self, "BSGValuePredictor", ValuePredictor, w, min_fr)
        self.action_model = action_model
        self.value_model = value_model
        self.action_input_adapter = action_input_adapter
        self.value_input_adapter = value_input_adapter

    def solve_from_env(self, env):        
        block_data = self.process_block_data(env.get_block_data())

        enc_data = self.action_input_adapter.enc_2_vec(block_data)
        enc_data = tuple(torch.from_numpy(data).unsqueeze(0) for data in enc_data)
        
        with torch.no_grad():
            enc_data_base = self.action_model.encode(*enc_data)

            while not env.is_finished():
                space_data_batch = self.process_space_data_batch(env.get_space_data_batch())
                pblock_data_batch = self.process_pblock_data_batch(env.get_pblock_data_batch())
                action_data_batch = self.process_action_data_batch(env.get_action_data_batch())

                batch_size = len(space_data_batch)

                enc_data_batch = tuple(
                    data.expand(batch_size, *data.shape[1:]) 
                    for data in enc_data_base
                )

                list_of_dec_tuples = [
                    self.action_input_adapter.dec_2_vec(block_data, space_data, pblock_data, action_data)
                    for action_data, pblock_data, space_data in zip(action_data_batch, pblock_data_batch, space_data_batch)
                ]
                
                dec_data_batch = tuple(
                    torch.from_numpy(np.stack(componentes))
                    for componentes in zip(*list_of_dec_tuples)
                )

                output = self.action_model.decode(*enc_data_batch, *dec_data_batch)

                ### Selección de acciones
                topk_values, topk_indices = output.topk(min(self.w, output.size(1)), dim=1)

                topk_indices_list = topk_indices.cpu().tolist()
                topk_values_list = topk_values.cpu().tolist()

                selected_blocks_batch = [
                    [
                        actions_disponibles[idx].block_id  # <--- Accedemos directo al atributo del objeto Action
                        for idx, score in zip(indices_elegidos, valores_scores) 
                        if score > -1e9
                    ]
                    for actions_disponibles, indices_elegidos, valores_scores in zip(action_data_batch, topk_indices_list, topk_values_list)
                ]

                env.expand(selected_blocks_batch)

                ### Evaluación greedy
                space_data_batch = self.process_space_data_batch(env.get_succ_space_data_batch())
                pblock_data_batch = self.process_pblock_data_batch(env.get_succ_pblock_data_batch())
                action_data_batch = self.process_action_data_batch(env.get_succ_action_data_batch())

                batch_size = len(space_data_batch)

                enc_data_batch = tuple(
                    data.expand(batch_size, *data.shape[1:]) 
                    for data in enc_data_base
                )

                list_of_dec_tuples = [
                    self.value_input_adapter.dec_2_vec(block_data, space_data, pblock_data, action_data)
                    for action_data, pblock_data, space_data in zip(action_data_batch, pblock_data_batch, space_data_batch)
                ]

                dec_data_batch = tuple(
                    torch.from_numpy(np.stack(componentes))
                    for componentes in zip(*list_of_dec_tuples)
                )

                output = self.value_model.decode(*enc_data_batch, *dec_data_batch)

                k = min(self.w, len(output))
                _, indices = torch.topk(output, k=k, largest=False)
                indices_list = indices.cpu().tolist()
                env.prune(indices_list)

        return env.best_volume * 100, env.final_time
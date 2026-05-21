from bsm_engine import BSM_VCS
from solvers.bs.bsm_solver import BSMSolver
from solvers.env_solver import EnvSolver
import torch
import numpy as np


class BSM_VCS_Solver(BSMSolver, EnvSolver):
    def __init__(self, model, input_adapter,w, min_fr):
        BSMSolver.__init__(self, "BSGValuePredictor", BSM_VCS, w, min_fr)
        self.model = model
        self.input_adapter = input_adapter

    def solve_from_env(self, env):        
        block_data = self.process_block_data(env.get_block_data())

        enc_data = self.input_adapter.enc_2_vec(block_data)
        enc_data = tuple(torch.as_tensor(data).unsqueeze(0) for data in enc_data)
        
        with torch.no_grad():
            enc_data_base = self.model.encode(*enc_data)

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
                    self.input_adapter.dec_2_vec(block_data, space_data, pblock_data, action_data)
                    for action_data, pblock_data, space_data in zip(action_data_batch, pblock_data_batch, space_data_batch)
                ]
                
                dec_data_batch = tuple(
                    torch.from_numpy(np.stack(componentes))
                    for componentes in zip(*list_of_dec_tuples)
                )

                output = self.model.decode(*enc_data_batch, *dec_data_batch)

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

                env.transition(selected_blocks_batch)

        return env.best_volume * 100, env.final_time
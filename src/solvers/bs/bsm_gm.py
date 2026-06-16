from bsm_engine import BSM_GM
from solvers.bs.bsm_solver import BSMSolver
from solvers.env_solver import EnvSolver
import torch
import numpy as np

class BSM_GM_Solver(BSMSolver, EnvSolver):
    def __init__(self, model, input_adapter, w, min_fr):
        BSMSolver.__init__(self, "BSM-GM", BSM_GM, w, min_fr)
        self.model = model
        self.input_adapter = input_adapter

    def solve_from_env(self, env):        
        block_data = self.process_block_data(env.get_block_data())

        enc_data = self.input_adapter.enc_2_vec(block_data)
        enc_data = tuple(torch.as_tensor(data).unsqueeze(0) for data in enc_data)

        with torch.no_grad():
            enc_data_base = self.model.encode(*enc_data)

            while not env.is_finished():
                space_data_batch = self.process_space_data_batch(env.get_space_data_batch_expand())
                pblock_data_batch = self.process_pblock_data_batch(env.get_pblock_data_batch_expand())
                action_data_batch = self.process_action_data_batch(env.get_action_data_batch_expand())

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

                env.expand(selected_blocks_batch)

                while not env.is_greedy_finished():
                    space_data_batch = self.process_space_data_batch(env.get_space_data_batch_greedy())
                    pblock_data_batch = self.process_pblock_data_batch(env.get_pblock_data_batch_greedy())
                    action_data_batch = self.process_action_data_batch(env.get_action_data_batch_greedy())

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

                    best_action_indices = output.argmax(dim=1)

                    selected_blocks = [
                        actions_disponibles[idx].block_id
                        for actions_disponibles, idx in zip(action_data_batch, best_action_indices)
                    ]

                    env.greedy_step(selected_blocks)
            
            return env.best_volume * 100, env.final_time


'''
        with torch.no_grad():
            memory = self.model.encode(block_features)

            while not bsm.is_bsm_finished():
                batch_data_bsm = bsm.get_batch_dict_bsm()
                
                action_blocks_batch = torch.from_numpy(batch_data_bsm["act_blocks"]).to(dtype=torch.int32)
                action_features_batch = torch.from_numpy(batch_data_bsm["act_feats"]).to(dtype=torch.float32)
                placed_blocks_batch = torch.from_numpy(batch_data_bsm["pl_blocks"]).to(dtype=torch.int32)
                placed_features_batch = torch.from_numpy(batch_data_bsm["pl_feats"]).to(dtype=torch.float32)
                space_features_batch = torch.from_numpy(batch_data_bsm["sp_feats"]).to(dtype=torch.float32)     

                B = action_blocks_batch.shape[0]
                curr_memory = memory.expand(B, -1, -1)
                output = self.model.decode(curr_memory, action_blocks_batch, action_features_batch,
                                         placed_blocks_batch, placed_features_batch, space_features_batch)
                    
                topk_values, topk_indices = output.topk(min(bsm.w, output.size(1)), dim=1)
                selected_action_blocks_batch = [
                    row_idx[row_val > -1e9].tolist() 
                    for row_idx, row_val in zip(action_blocks_batch.gather(1, topk_indices), topk_values)
                ]

                bsm.transition_bsm(selected_action_blocks_batch)
                               
                while not bsm.is_greedy_finished():
                    batch_data_gr = bsm.get_batch_dict_greedy()
                    
                    action_blocks_batch = torch.from_numpy(batch_data_gr["act_blocks"]).to(dtype=torch.int32)
                    action_features_batch = torch.from_numpy(batch_data_gr["act_feats"]).to(dtype=torch.float32)
                    placed_blocks_batch = torch.from_numpy(batch_data_gr["pl_blocks"]).to(dtype=torch.int32)
                    placed_features_batch = torch.from_numpy(batch_data_gr["pl_feats"]).to(dtype=torch.float32)
                    space_features_batch = torch.from_numpy(batch_data_gr["sp_feats"]).to(dtype=torch.float32)
                    
                    B = action_blocks_batch.shape[0]
                    curr_memory = memory.expand(B, -1, -1)
                    output = self.model.decode(curr_memory, action_blocks_batch, action_features_batch,
                                             placed_blocks_batch, placed_features_batch, space_features_batch)
                    
                    best_action_indices = output.argmax(dim=1)
                    rows = torch.arange(B)
                    selected_blocks = action_blocks_batch[rows, best_action_indices].tolist()  
                    bsm.transition_greedy(selected_blocks)

        return bsm.best_volume * 100, bsm.final_time
'''
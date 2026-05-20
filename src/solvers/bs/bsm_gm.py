from models.base.transformer import Transformer
from bsm_engine import BSM_GM
from solvers.bs.bsm_solver import BSM_Solver
import torch

class BSM_GM_Solver(BSM_Solver):
    def __init__(self, model: Transformer):
        super().__init__(model, "BSM-GM", BSM_GM)

    def solve_from_env(self, bsm: BSM_GM):        
        raw_bl_feats = bsm.get_block_features()
        block_features = torch.from_numpy(raw_bl_feats).unsqueeze(0)
        
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
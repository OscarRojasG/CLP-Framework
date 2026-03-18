from models.base.transformer import Transformer
from settings import INSTANCE_FOLDER
from bsm_vcs_engine import BSM_VCS
from solvers.bs_solver import BS_Solver
import os
import torch

class BSM_VCS_Solver(BS_Solver):
    def __init__(self, model: Transformer, w):
        self.model = model
        self.w = w

    def solve(self, instance_file, instance_number):
        instance_file = str(INSTANCE_FOLDER / instance_file)
        
        if os.path.exists(instance_file) == False:
            raise Exception(f'El archivo de instancia {instance_file} no existe.')
        
        bsm = BSM_VCS(instance_file, instance_number, self.w)
        
        raw_bl_feats = bsm.get_block_features()
        block_features = torch.from_numpy(raw_bl_feats).unsqueeze(0)
        
        with torch.no_grad():
            memory = self.model.encode(block_features)

            while not bsm.is_finished():
                batch_data_bsm = bsm.get_batch_dict()
                
                action_blocks_batch = torch.from_numpy(batch_data_bsm["act_blocks"]).to(dtype=torch.int32)
                action_features_batch = torch.from_numpy(batch_data_bsm["act_feats"]).to(dtype=torch.float32)
                placed_blocks_batch = torch.from_numpy(batch_data_bsm["pl_blocks"]).to(dtype=torch.int32)
                placed_features_batch = torch.from_numpy(batch_data_bsm["pl_feats"]).to(dtype=torch.float32)
                space_features_batch = torch.from_numpy(batch_data_bsm["sp_feats"]).to(dtype=torch.float32)     

                B = action_blocks_batch.shape[0]
                curr_memory = memory.expand(B, -1, -1)
                output = self.model.decode(curr_memory, action_blocks_batch, action_features_batch,
                                         placed_blocks_batch, placed_features_batch, space_features_batch)
                    
                topk_values, topk_indices = output.topk(min(self.w, output.size(1)), dim=1)
                selected_action_blocks_batch = [
                    row_idx[row_val > -1e9].tolist() 
                    for row_idx, row_val in zip(action_blocks_batch.gather(1, topk_indices), topk_values)
                ]

                bsm.transition(selected_action_blocks_batch)

        return bsm.best_volume * 100
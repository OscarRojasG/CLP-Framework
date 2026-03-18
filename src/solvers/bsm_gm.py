from models.base.transformer import Transformer
from settings import INSTANCE_FOLDER
from bsm_gm_engine import BSM_GM
import os
import torch
import time

'''
class BSMSolver():
    def __init__(self, model: Transformer):
        self.model = model

    def solve(self, instance_file, instance_number, w: int) -> int:
        instance_file = str(INSTANCE_FOLDER / instance_file)
        
        if os.path.exists(instance_file) == False:
            raise Exception(f'El archivo de instancia {instance_file} no existe.')
        
        # Acumuladores para el diagnóstico
        total_cpp_to_py_time = 0.0
        total_py_to_torch_time = 0.0
        
        bsm = BSM_GM(instance_file, instance_number, w)
        
        raw_bl_feats = bsm.get_block_features()
        block_features = torch.from_numpy(raw_bl_feats).unsqueeze(0)
        
        with torch.no_grad():
            memory = self.model.encode(block_features)

            while not bsm.is_bsm_finished():
                # --- DIAGNÓSTICO BSM (INICIO) ---
                t0 = time.perf_counter()
                batch_data_bsm = bsm.get_full_batch_bsm()
                t1 = time.perf_counter()
                
                action_blocks_batch = torch.from_numpy(batch_data_bsm["act_blocks"]).to(dtype=torch.int32)
                action_features_batch = torch.from_numpy(batch_data_bsm["act_feats"]).to(dtype=torch.float32)
                placed_blocks_batch = torch.from_numpy(batch_data_bsm["pl_blocks"]).to(dtype=torch.int32)
                placed_features_batch = torch.from_numpy(batch_data_bsm["pl_feats"]).to(dtype=torch.float32)
                space_features_batch = torch.from_numpy(batch_data_bsm["sp_feats"]).to(dtype=torch.float32)
                t2 = time.perf_counter()
                
                total_cpp_to_py_time += (t1 - t0)
                total_py_to_torch_time += (t2 - t1)
                # --- DIAGNÓSTICO BSM (FIN) ---

                B = action_blocks_batch.shape[0]
                curr_memory = memory.expand(B, -1, -1)
                output = self.model.decode(curr_memory, action_blocks_batch, action_features_batch,
                                         placed_blocks_batch, placed_features_batch, space_features_batch)
                    
                topk_values, topk_indices = output.topk(min(w, output.size(1)), dim=1)
                selected_action_blocks_batch = [
                    row_idx[row_val > -1e9].tolist() 
                    for row_idx, row_val in zip(action_blocks_batch.gather(1, topk_indices), topk_values)
                ]

                bsm.transition_bsm(selected_action_blocks_batch)
                               
                while not bsm.is_greedy_finished():
                    # --- DIAGNÓSTICO GREEDY (INICIO) ---
                    t0_g = time.perf_counter()
                    batch_data_gr = bsm.get_full_batch_greedy()
                    t1_g = time.perf_counter()
                    
                    action_blocks_batch = torch.from_numpy(batch_data_gr["act_blocks"]).to(dtype=torch.int32)
                    action_features_batch = torch.from_numpy(batch_data_gr["act_feats"]).to(dtype=torch.float32)
                    placed_blocks_batch = torch.from_numpy(batch_data_gr["pl_blocks"]).to(dtype=torch.int32)
                    placed_features_batch = torch.from_numpy(batch_data_gr["pl_feats"]).to(dtype=torch.float32)
                    space_features_batch = torch.from_numpy(batch_data_gr["sp_feats"]).to(dtype=torch.float32)
                    t2_g = time.perf_counter()
                    
                    total_cpp_to_py_time += (t1_g - t0_g)
                    total_py_to_torch_time += (t2_g - t1_g)
                    # --- DIAGNÓSTICO GREEDY (FIN) ---
                    
                    B = action_blocks_batch.shape[0]
                    curr_memory = memory.expand(B, -1, -1)
                    output = self.model.decode(curr_memory, action_blocks_batch, action_features_batch,
                                             placed_blocks_batch, placed_features_batch, space_features_batch)
                    
                    best_action_indices = output.argmax(dim=1)
                    rows = torch.arange(B)
                    selected_blocks = action_blocks_batch[rows, best_action_indices].tolist()  
                    bsm.transition_greedy(selected_blocks)
            
        print("-" * 30)
        print(f"RESULTADOS DEL DIAGNÓSTICO:")
        print(f"C++ a Listas Python (Pybind11): {total_cpp_to_py_time:.4f}s")
        print(f"Listas Python a Tensors (Torch): {total_py_to_torch_time:.4f}s")
        print(f"Total acumulado: {total_cpp_to_py_time + total_py_to_torch_time:.4f}s")
        print("-" * 30)

        return bsm.best_volume * 100
'''

class BSM_GM_Solver():
    def __init__(self, model: Transformer):
        self.model = model

    def solve(self, instance_file, instance_number, w: int) -> int:
        instance_file = str(INSTANCE_FOLDER / instance_file)
        
        if os.path.exists(instance_file) == False:
            raise Exception(f'El archivo de instancia {instance_file} no existe.')
        
        bsm = BSM_GM(instance_file, instance_number, w)
        
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
                    
                topk_values, topk_indices = output.topk(min(w, output.size(1)), dim=1)
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

        return bsm.best_volume * 100
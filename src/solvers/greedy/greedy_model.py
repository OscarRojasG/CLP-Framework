import torch
from bsm_engine import GreedyModel
from solvers.greedy.greedy_solver import GreedySolver
from settings import INSTANCE_FOLDER
import os


class GreedyModelSolver(GreedySolver): 
    def __init__(self, model, w):
        super().__init__("GreedyModel")
        self.model = model
        self.w = w
        pass
     
    def solve(self, instance_file, instance_number, min_fr):
        instance_file = str(INSTANCE_FOLDER / instance_file) 
        
        if os.path.exists(instance_file) == False:
            raise Exception(f'El archivo de instancia {instance_file} no existe.')
        
        env = GreedyModel(instance_file, instance_number, self.w, min_fr)
        
        raw_bl_feats = env.get_block_features()
        block_features = torch.from_numpy(raw_bl_feats).unsqueeze(0)
        
        with torch.no_grad():
            memory = self.model.encode(block_features)

            while not env.is_finished():
                data = env.get_dict()
                
                action_blocks = torch.from_numpy(data["act_blocks"]).to(dtype=torch.int32).unsqueeze(0)
                action_features = torch.from_numpy(data["act_feats"]).to(dtype=torch.float32).unsqueeze(0)
                placed_blocks = torch.from_numpy(data["pl_blocks"]).to(dtype=torch.int32).unsqueeze(0)
                placed_features = torch.from_numpy(data["pl_feats"]).to(dtype=torch.float32).unsqueeze(0)
                space_features = torch.from_numpy(data["sp_feats"]).to(dtype=torch.float32).unsqueeze(0)
                biases = torch.from_numpy(data["biases"]).to(dtype=torch.float32).unsqueeze(0)

                if self.model.biased:
                    output = self.model.decode(memory, action_blocks, action_features, placed_blocks, placed_features, space_features, biases)
                else:
                    output = self.model.decode(memory, action_blocks, action_features, placed_blocks, placed_features, space_features)

                best_index = output.argmax(dim=1).item()
                selected_block = action_blocks[0, best_index].item()
                
                env.transition(selected_block)

        vol = env.volume * 100
        time = env.final_time
        del env
        return vol, time
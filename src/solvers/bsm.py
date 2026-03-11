from envs.bsm_env import BSMEnvironment, BSM
from models.base.transformer import Transformer
import torch

class BSMSolver():
    def __init__(self, model: Transformer):
        self.model = model
        self.env = BSMEnvironment()

    def solve(self, instance_file, instance_number, w: int) -> int:
        bsm = self.env.init(instance_file, instance_number, w)
        volume = self._solve(w, bsm)
        bsm.close()
        return volume

    def _solve(self, w: int, bsm: BSM) -> int:
        block_features = torch.as_tensor(bsm.block_features.copy(), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            memory = self.model.encode(block_features)

            while True:
                B = bsm.action_blocks_batch.shape[0]

                # Expandimos la memoria si es necesario
                curr_memory = memory.expand(B, -1, -1)
                    
                # Predecir las mejores acciones
                output = self.model.decode(
                    curr_memory,
                    bsm.action_blocks_batch,
                    bsm.action_features_batch,
                    bsm.placed_blocks_batch,
                    bsm.placed_features_batch,
                    bsm.space_features_batch
                )
                    
                topk_values, topk_indices = output.topk(min(w, output.size(1)), dim=1)
                selected_action_blocks_batch = [
                    row_idx[row_val > -1e9].tolist() 
                    for row_idx, row_val in zip(bsm.action_blocks_batch.gather(1, topk_indices), topk_values)
                ]

                # Generar sucesores (w2)
                greedy_process = self.env.next(bsm, selected_action_blocks_batch)
                
                # Aplicar greedy a cada sucesor
                while greedy_process.finished == False:
                    B = greedy_process.bsm.num_states
                    
                    # Expandimos la memoria si es necesario
                    curr_memory = memory.expand(B, -1, -1)
                        
                    # Predecir las mejores acciones
                    output = self.model.decode(
                        curr_memory,
                        greedy_process.action_blocks_batch,
                        greedy_process.action_features_batch,
                        greedy_process.placed_blocks_batch,
                        greedy_process.placed_features_batch,
                        greedy_process.space_features_batch
                    )
                    
                    # Seleccionamos la mejor acción por cada estado en el batch
                    best_action_indices = output.argmax(dim=1)
                    
                    rows = torch.arange(B)
                    selected_blocks = greedy_process.action_blocks_batch[rows, best_action_indices].tolist()
                    
                    greedy_process.transition(selected_blocks)
                    
                if bsm.num_states == 0:
                    break
                
                # Actualizar generación actual (w)
                bsm.update()
            
        return bsm.volume_ratio * 100
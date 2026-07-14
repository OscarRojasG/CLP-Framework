from solvers.beam_search.bsm_solver import BSMSolver
from solvers.env_solver import EnvSolver
import torch


class BSG_ModelVCS_Solver(BSMSolver, EnvSolver):
    def __init__(self, model, input_adapter, w, min_fr, inference_mode=True):
        if inference_mode:
            from envs.bsm_engine_inf import BSM_VCS
        else:
            from envs.bsm_engine_dev import BSM_VCS
            
        super().__init__("BSM-VCS", BSM_VCS, w, min_fr, input_adapter, inference_mode)
        self.model = model
        self.input_adapter = input_adapter

    def solve_from_env(self, env):        
        device = next(self.model.parameters()).device
        
        # 1. Preparar Encoder
        # Si no es inferencia, necesitamos block_data para el adapter
        block_data = self.process_block_data(env.get_block_data()) if not self.inference_mode else None
        enc_tensors = self.get_enc_inputs(env, block_data, device)
        
        with torch.inference_mode() if self.inference_mode else torch.no_grad():
            enc_data_base = self.model.encode(*enc_tensors)

            while not env.is_finished():
                # 2. Preparar Decoder
                dec_tensors, action_data_batch = self.get_dec_inputs_batch(env, block_data, device)
                
                # BATCH SIZE DINÁMICO
                # dec_tensors[0] tiene forma [B, max_actions]
                current_batch_size = dec_tensors[0].shape[0]

                # MEMORY EXPANSION (Independiente del modo)
                # enc_data_base viene de [1, N, D] -> expandimos a [B, N, D]
                # Esto es una 'view' (zero-copy), muy rápido.
                memory = [m.expand(current_batch_size, *m.shape[1:]) for m in enc_data_base]

                # 3. Decodificación
                output = self.model.decode(*memory, *dec_tensors)

                # 3. Selección de acciones
                topk_values, topk_indices = output.topk(min(self.w, output.size(1)), dim=1)

                if self.inference_mode:
                    action_ids_tensor = dec_tensors[0] # [B, max_actions]
                    indices = topk_indices # [B, w]
                    
                    # gather es una operación de PyTorch que hace el mapeo por ti
                    # Mapea los índices de top-k a los IDs físicos que están en action_ids_tensor
                    selected_ids_tensor = torch.gather(action_ids_tensor, 1, indices)
                    
                    # Convertimos a lista de listas para C++
                    # El -1 es el valor de padding que definimos
                    selected_ids_batch = []
                    for row in selected_ids_tensor.cpu().tolist():
                        selected_ids_batch.append([int(x) for x in row if x != -1])
                        
                    env.transition(selected_ids_batch)
                else:
                    # Modo Dev: Mapeamos índices a block_id como lo hacías originalmente
                    topk_indices_list = topk_indices.cpu().tolist()
                    topk_values_list = topk_values.cpu().tolist()
                    
                    selected_blocks_batch = [
                        [actions[idx].block_id for idx, score in zip(indices, values) if score > -1e9]
                        for actions, indices, values in zip(action_data_batch, topk_indices_list, topk_values_list)
                    ]
                    env.transition(selected_blocks_batch)

        return env.best_volume * 100, env.final_time
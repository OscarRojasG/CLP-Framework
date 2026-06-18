from solvers.bs.bsm_solver import BSMSolver
from solvers.env_solver import EnvSolver
import torch
import numpy as np

class BSM_GM_Solver(BSMSolver, EnvSolver):
    def __init__(self, model, input_adapter, w, min_fr, inference_mode=False):
        # Seleccionamos dinámicamente la versión del entorno compilada
        if inference_mode:
            from envs.bsm_engine_inf import BSM_GM
        else:
            from envs.bsm_engine_dev import BSM_GM
            
        super().__init__("BSM-GM", BSM_GM, w, min_fr, input_adapter, inference_mode)
        self.model = model
        self.input_adapter = input_adapter

    def solve_from_env(self, env):        
        device = next(self.model.parameters()).device
        
        # 1. Preparar Encoder (Se calcula una sola vez)
        block_data = self.process_block_data(env.get_block_data()) if not self.inference_mode else None
        enc_tensors = self.get_enc_inputs(env, block_data, device)

        with torch.inference_mode() if self.inference_mode else torch.no_grad():
            # Codificación base estática
            enc_data_base = self.model.encode(*enc_tensors)

            while not env.is_finished():
                
                # ==========================================
                # FASE 1: EXPAND (Beam Search Branching)
                # ==========================================
                # Usamos la nueva función modular
                dec_tensors_expand, action_data_expand = self.get_dec_inputs_batch_expand(env, block_data, device)
                
                # Zero-copy memory expansion
                current_batch_size = dec_tensors_expand[0].shape[0]
                memory_expand = [m.expand(current_batch_size, *m.shape[1:]) for m in enc_data_base]

                # Decodificación y Selección Expand
                output_expand = self.model.decode(*memory_expand, *dec_tensors_expand)
                topk_values, topk_indices = output_expand.topk(min(self.w, output_expand.size(1)), dim=1)

                if self.inference_mode:
                    action_ids_tensor = dec_tensors_expand[0]
                    selected_ids_tensor = torch.gather(action_ids_tensor, 1, topk_indices)
                    
                    selected_ids_batch = []
                    for row in selected_ids_tensor.cpu().tolist():
                        selected_ids_batch.append([int(x) for x in row if x != -1])
                        
                    env.expand(selected_ids_batch)
                else:
                    topk_indices_list = topk_indices.cpu().tolist()
                    topk_values_list = topk_values.cpu().tolist()
                    
                    selected_blocks_batch = [
                        [actions[idx].block_id for idx, score in zip(indices, values) if score > -1e9]
                        for actions, indices, values in zip(action_data_expand, topk_indices_list, topk_values_list)
                    ]
                    env.expand(selected_blocks_batch)

                # ==========================================
                # FASE 2: GREEDY LOOP (Rollout hasta las hojas)
                # ==========================================
                while not env.is_greedy_finished():
                    # Usamos la nueva función modular
                    dec_tensors_greedy, action_data_greedy = self.get_dec_inputs_batch_greedy(env, block_data, device)
                    
                    # Zero-copy memory expansion para el lote greedy
                    current_batch_size = dec_tensors_greedy[0].shape[0]
                    memory_greedy = [m.expand(current_batch_size, *m.shape[1:]) for m in enc_data_base]

                    # Decodificación y Selección Greedy (Argmax)
                    output_greedy = self.model.decode(*memory_greedy, *dec_tensors_greedy)
                    best_indices = output_greedy.argmax(dim=1, keepdim=True)

                    if self.inference_mode:
                        action_ids_tensor_greedy = dec_tensors_greedy[0]
                        selected_ids_tensor = torch.gather(action_ids_tensor_greedy, 1, best_indices)
                        
                        selected_blocks = [int(x[0]) for x in selected_ids_tensor.cpu().tolist()]
                        env.greedy_step(selected_blocks)
                    else:
                        best_indices_list = best_indices.squeeze(1).cpu().tolist()
                        
                        selected_blocks = [
                            actions[idx].block_id
                            for actions, idx in zip(action_data_greedy, best_indices_list)
                        ]
                        env.greedy_step(selected_blocks)

        return env.best_volume * 100, env.final_time
    
    def get_dec_inputs_batch_expand(self, env, block_data_cache, device):
        """Prepara los tensores del decoder para la fase de expansión (Beam Search)"""
        if self.inference_mode:
            # Modo Inferencia: Obtenemos tensores directamente desde C++
            dec_numpy = env.get_dec_data_batch_expand()
            dec_tensors = tuple(torch.from_numpy(data).to(device) for data in dec_numpy)
            return dec_tensors, None
        else:
            # Modo Dev: Pipeline original con proceso de batches
            space_data_batch = self.process_space_data_batch(env.get_space_data_batch_expand())
            pblock_data_batch = self.process_pblock_data_batch(env.get_pblock_data_batch_expand())
            action_data_batch = self.process_action_data_batch(env.get_action_data_batch_expand())

            # Empaquetado manual para el modo dev
            list_of_dec_tuples = [
                self.input_adapter.dec_2_vec(block_data_cache, s, p, a)
                for a, p, s in zip(action_data_batch, pblock_data_batch, space_data_batch)
            ]
            
            dec_tensors = tuple(
                torch.from_numpy(np.stack(componentes)).to(device)
                for componentes in zip(*list_of_dec_tuples)
            )
            return dec_tensors, action_data_batch


    def get_dec_inputs_batch_greedy(self, env, block_data_cache, device):
        """Prepara los tensores del decoder para la fase de simulación voraz (Greedy)"""
        if self.inference_mode:
            # Modo Inferencia: Obtenemos tensores directamente desde C++
            dec_numpy = env.get_dec_data_batch_greedy()
            dec_tensors = tuple(torch.from_numpy(data).to(device) for data in dec_numpy)
            return dec_tensors, None
        else:
            # Modo Dev: Pipeline original con proceso de batches
            space_data_batch = self.process_space_data_batch(env.get_space_data_batch_greedy())
            pblock_data_batch = self.process_pblock_data_batch(env.get_pblock_data_batch_greedy())
            action_data_batch = self.process_action_data_batch(env.get_action_data_batch_greedy())

            # Empaquetado manual para el modo dev
            list_of_dec_tuples = [
                self.input_adapter.dec_2_vec(block_data_cache, s, p, a)
                for a, p, s in zip(action_data_batch, pblock_data_batch, space_data_batch)
            ]
            
            dec_tensors = tuple(
                torch.from_numpy(np.stack(componentes)).to(device)
                for componentes in zip(*list_of_dec_tuples)
            )
            return dec_tensors, action_data_batch
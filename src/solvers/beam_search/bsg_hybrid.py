from solvers.beam_search.bsm_solver import BSMSolver
from solvers.env_solver import EnvSolver
import torch


class BSG_Hybrid_Solver(BSMSolver, EnvSolver):
    def __init__(self, model, input_adapter, w, min_fr, fixed_w=True):
        from envs.clp_engine import BSG_Hybrid
        super().__init__("BSM-VCS", BSG_Hybrid, w, min_fr, input_adapter)
        self.model = model
        self.input_adapter = input_adapter
        self.fixed_w = fixed_w

    def solve_from_env(self, env):        
        device = next(self.model.parameters()).device
        
        # 1. Encoder
        enc_tensors = self.get_enc_inputs(env, device)

        with torch.inference_mode():
            enc_data_base = self.model.encode(*enc_tensors)

            while not env.is_finished():
                # 2. Preparar Decoder
                dec_tensors = self.get_dec_inputs_batch(env, device)
                
                # BATCH SIZE DINÁMICO
                # dec_tensors[0] tiene forma [B, max_actions]
                current_batch_size = dec_tensors[0].shape[0]

                # MEMORY EXPANSION
                # enc_data_base viene de [1, N, D] -> expandimos a [B, N, D]
                # Esto es una 'view' (zero-copy), muy rápido.
                memory = [m.expand(current_batch_size, *m.shape[1:]) for m in enc_data_base]

                # 3. Decodificación
                output = self.model.decode(*memory, *dec_tensors)

                # 3. Selección de acciones
                if self.fixed_w:
                    topk_values, topk_indices = output.topk(min(self.w, output.size(1)), dim=1)
                else:
                    # Se reparten w*w hijos entre los nodos del batch
                    k = int(self.w * self.w / current_batch_size + 0.5)
                    topk_values, topk_indices = output.topk(min(k, output.size(1)), dim=1)

                # Mapea los índices de top-k a los IDs de bloque que están en dec_tensors[0] ([B, max_actions])
                selected_ids_tensor = torch.gather(dec_tensors[0], 1, topk_indices)

                # Lista de listas para C++, sin el padding (-1)
                selected_ids_batch = []
                for row in selected_ids_tensor.cpu().tolist():
                    selected_ids_batch.append([int(x) for x in row if x != -1])

                env.transition(selected_ids_batch)

        return env.best_volume * 100, env.final_time
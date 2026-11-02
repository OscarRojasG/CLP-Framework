from solvers.beam_search.bsg_model_vcs import BSG_ModelVCS_Solver
from settings import INSTANCE_FOLDER
import torch
import math
import os


class DE_BSG_ModelVCS_Solver(BSG_ModelVCS_Solver):
    """
    Replica DoubleEffort de BSG_CLP (-t): ejecuta la búsqueda completa desde la raíz,
    aumenta w en un factor sqrt(2) y repite hasta alcanzar el tiempo límite.
    La mejor solución y el tiempo transcurrido se conservan entre ejecuciones.
    En cada ejecución max_actions = w*w.
    Solo modo inferencia. input_adapter no se usa (puede ser None); se mantiene
    en la firma por compatibilidad con fast_eval.
    """
    INITIAL_W = 4

    def __init__(self, model, input_adapter, timelimit, min_fr):
        super().__init__(model, input_adapter, self.INITIAL_W, min_fr)
        self.name = "DE-BSM-VCS"
        self.timelimit = timelimit

    def load_env(self, file, instance_number):
        file = str(INSTANCE_FOLDER / file)

        if not os.path.exists(file):
            raise Exception(f'El archivo de instancia {file} no existe.')

        # max_blocks y max_pblocks no se usan en inferencia; max_actions se fija en cada corrida
        return self.solver_module(file, instance_number, self.w, 0, self.w * self.w, 0, self.min_fr)

    def solve(self, file, instance_number):
        self.w = self.INITIAL_W
        env = self.load_env(file, instance_number)
        env.timelimit = self.timelimit

        # El encoder solo depende de los bloques, por lo que se calcula una vez para todas las corridas
        device = next(self.model.parameters()).device
        enc_data_base = self._encode(env, device)

        while env.get_elapsed_time() <= self.timelimit:
            env.w = self.w
            env.max_actions = self.w * self.w
            env.reset()
            self._search(env, enc_data_base, device)

            # Igual que BSG::double_effort
            self.w = int(self.w * math.sqrt(2) + 0.5) if self.w > 1 else 2

        vol, time = env.best_volume * 100, env.final_time
        del env
        return vol, time

    def _encode(self, env, device):
        enc_tensors = self.get_enc_inputs(env, None, device)

        with torch.inference_mode():
            return self.model.encode(*enc_tensors)

    # Mismo bucle que BSG_ModelVCS_Solver.solve_from_env (modo inferencia), sin el encoder
    def _search(self, env, enc_data_base, device):
        with torch.inference_mode():
            while not env.is_finished():
                dec_tensors, _ = self.get_dec_inputs_batch(env, None, device)

                # dec_tensors[0] tiene forma [B, max_actions]
                current_batch_size = dec_tensors[0].shape[0]

                # enc_data_base viene de [1, N, D] -> expandimos a [B, N, D]
                memory = [m.expand(current_batch_size, *m.shape[1:]) for m in enc_data_base]

                output = self.model.decode(*memory, *dec_tensors)

                topk_values, topk_indices = output.topk(min(self.w, output.size(1)), dim=1)

                # Mapea los índices de top-k a los IDs que están en dec_tensors[0]
                selected_ids_tensor = torch.gather(dec_tensors[0], 1, topk_indices)

                # El -1 es el valor de padding
                selected_ids_batch = []
                for row in selected_ids_tensor.cpu().tolist():
                    selected_ids_batch.append([int(x) for x in row if x != -1])

                env.transition(selected_ids_batch)

import numpy as np
from data.adapters.output.output_adapter import OutputAdapter
from data.objects import Action

class RankingAdapter(OutputAdapter):
    def __init__(self, max_actions=64):
        super().__init__({
            "Y": np.float32
        })
        self.max_actions = max_actions

    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: list):
        # 1. Asegurar el vector base en float32
        y_vector = np.zeros(self.max_actions, dtype=np.float32)

        # 2. Convertir la lista greedy completa a un array de numpy
        # Confiamos en que greedy siempre contiene los 64 scores correspondientes
        valid_greedy = np.array(greedy[:self.max_actions], dtype=np.float32)

        # 3. Calcular ranking denso para manejo robusto de empates
        # unique_vals se ordena de menor a mayor por defecto
        unique_vals, inverse_indices = np.unique(valid_greedy, return_inverse=True)
        
        # Asignamos el rango (los peores reciben 1, los mejores reciben el rango más alto)
        dense_ranks = inverse_indices + 1
        
        y_vector[:] = dense_ranks

        return (y_vector, )
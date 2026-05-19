import numpy as np
from data.adapters.output.output_adapter import OutputAdapter
from data.objects import Action

class ActionAdapter(OutputAdapter):
    def __init__(self, max_actions):
        super().__init__({
            "Y": np.int32
        })
        self.max_actions = max_actions

    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: float):
        # 1. Pre-asignar vector de ceros con el largo máximo
        y_vector = np.zeros(self.max_actions, dtype=np.int32)
        
        # 2. Activar el índice del bloque seleccionado
        for i, action in enumerate(actions):
            if action.block_id == selected_block:
                y_vector[i] = 1
        
                # Retornar el diccionario con el arreglo numpy
                return (y_vector, )
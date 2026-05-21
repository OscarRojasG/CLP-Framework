import numpy as np
from data.adapters.output.output_adapter import OutputAdapter
from data.objects import Action

class ActionAdapter(OutputAdapter):
    def __init__(self, max_actions):
        super().__init__({
            "Y": np.int32
        })
        self.max_actions = max_actions

    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: list):
        # 1. Pre-asignar vector de ceros con el largo máximo
        y_vector = np.zeros(self.max_actions, dtype=np.int32)
        y_vector[np.argmax(greedy)] = 1
        
        return (y_vector, )
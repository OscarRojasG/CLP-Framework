import numpy as np
from data.adapters.output.output_adapter import OutputAdapter
from data.objects import Action

class ActionAdapter(OutputAdapter):
    def __init__(self, max_actions: int):
        super().__init__({
            "Y": np.float32
        })
        self.max_actions = max_actions

    """
    Retorna: (y_vector,)
    Features: Vector one-hot con 1 en el índice de la mejor acción ('selected_block').
    El label smoothing se aplica en la función de pérdida (CrossEntropyLoss).
    """
    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: list):
        y_vector = np.zeros(self.max_actions, dtype=np.float32)
        
        for i, action in enumerate(actions):
            if action.block_id == selected_block:
                y_vector[i] = 1.0
                break
        
        return (y_vector, )
import numpy as np
from data.adapters.output.output_adapter import OutputAdapter
from data.objects import Action

class CSAdapter(OutputAdapter):
    def __init__(self, max_actions):
        super().__init__({
            "Y": np.float32
        })
        self.max_actions = max_actions

    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: list):
        y_vector = np.zeros(self.max_actions, dtype=np.float32)
        for i, action in enumerate(actions):
            y_vector[i] = action.cs + (action.loss if action.loss > 0 else 0)
            if action.cs == 0 and action.loss <= 0:
                print('no debiera pasar')

        y_vector = y_vector / y_vector.sum()

        return (y_vector, )
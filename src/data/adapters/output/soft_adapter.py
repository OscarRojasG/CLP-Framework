import numpy as np
from data.adapters.output.output_adapter import OutputAdapter
from data.objects import Action

class SoftAdapter(OutputAdapter):
    def __init__(self, max_actions):
        super().__init__({
            "Y": np.float32
        })
        self.max_actions = max_actions

    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: list):
        y_vector = np.zeros(self.max_actions, dtype=np.float32)
        n_actions = len(actions)
        if n_actions == 0:
            return (y_vector,)

        tau = 0.007
        g = np.array(greedy[:n_actions], dtype=np.float32)
        
        # Estabilidad numérica + softmax con temperatura
        g_shifted = g - g.max()
        soft = np.exp(g_shifted / tau)
        y_vector[:n_actions] = soft / soft.sum()

        return (y_vector,)
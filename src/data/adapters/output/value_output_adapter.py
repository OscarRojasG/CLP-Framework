import numpy as np
from data.adapters.output.output_adapter import OutputAdapter
from data.objects import Action

class ValueOutputAdapter(OutputAdapter):
    def __init__(self):
        super().__init__({
            "Y": np.float32
        })

    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: float):
        return (100 * (1-greedy), )
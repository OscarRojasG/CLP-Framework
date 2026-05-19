import numpy as np
from data.adapters.output.output_adapter import OutputAdapter
from data.objects import Action

class ValueAdapter(OutputAdapter):
    def __init__(self):
        super().__init__({
            "Y": np.float32
        })

    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: float):
        for action in actions:
            if action.block_id == selected_block:
                greedy = action.greedy
                return (np.log(1-greedy), )
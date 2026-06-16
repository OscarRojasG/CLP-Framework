import numpy as np
from data.adapters.output.output_adapter import OutputAdapter
from data.objects import Action

class ActionAdapter(OutputAdapter):
    def __init__(self, max_actions: int, label_smoothing: float = 0.0):
        super().__init__({
            "Y": np.float32  # Siempre lo mantenemos como float
        })
        self.max_actions = max_actions
        self.label_smoothing = label_smoothing

    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: list):
        # La matemática funciona sola: si smoothing es 0.0, off_value = 0.0 y on_value = 1.0
        off_value = self.label_smoothing / (self.max_actions - 1) if self.max_actions > 1 else 0.0
        on_value = 1.0 - self.label_smoothing
        
        # Inicializamos todo el vector con el off_value
        y_vector = np.full(self.max_actions, off_value, dtype=np.float32)
        
        for i, action in enumerate(actions):
            if action.block_id == selected_block:
                y_vector[i] = on_value
                break
        
        return (y_vector, )
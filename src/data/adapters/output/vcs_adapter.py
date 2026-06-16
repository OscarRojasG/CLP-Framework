import numpy as np
from data.adapters.output.output_adapter import OutputAdapter
from data.objects import Action

class VCSAdapter(OutputAdapter):
    def __init__(self, max_actions):
        super().__init__({
            "Y": np.float32
        })
        self.max_actions = max_actions

    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: list):
        # Inicializar el vector con ceros
        y_vector = np.zeros(self.max_actions, dtype=np.float32)
        
        if not actions:
            return (y_vector, )

        # 1. Extraer los puntajes (scores) de las acciones disponibles
        scores = np.array([action.calc_vcs() for action in actions], dtype=np.float32)
        
        # 2. Normalizar a media 0, std 1 (basado en la lógica original)
        # Se suma 1e-8 para evitar división por cero si todos los scores son iguales
        scores_normalized = (scores - scores.mean()) / (scores.std() + 1e-8)
        
        # 3. Aplicar temperatura
        temperature = 1
        scaled_scores = scores_normalized / temperature
        
        # 4. Aplicar Softmax (con truco de estabilidad numérica restando el máximo)
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        soft_labels = exp_scores / exp_scores.sum()
        
        # 5. Asignar los soft labels calculados al vector final
        # Solo se sobreescriben los primeros índices correspondientes a las acciones reales.
        # El resto del vector (padding) permanecerá en 0.0
        for i, soft_label in enumerate(soft_labels):
            y_vector[i] = soft_label

        return (y_vector, )
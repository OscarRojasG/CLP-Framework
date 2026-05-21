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
        # 1. Inicialización limpia (La línea que mencionabas)
        y_vector = np.zeros(self.max_actions, dtype=np.float32)
        
        # 2. Convertir valores greedy a un array numpy para operar
        q_values = np.array(greedy, dtype=np.float32)
        
        # 3. Determinar K (tomar el mínimo entre lo disponible y tu límite)
        k = min(8, len(q_values))
        
        # 4. Obtener índices de los K valores más altos
        # argpartition es O(n), mucho más eficiente que sort para buscar top-k
        top_k_indices = np.argpartition(q_values, -k)[-k:]
        
        # 5. Aplicar Softmax con temperatura sobre los valores del Top-K
        top_k_values = q_values[top_k_indices]
        
        # Restar el máximo para estabilidad numérica antes de la exponencial
        temperature = 0.005
        exp_values = np.exp((top_k_values - np.max(top_k_values)) / temperature)
        probs = exp_values / np.sum(exp_values)
        
        # 6. Asignar las probabilidades normalizadas al vector final
        y_vector[top_k_indices] = probs
        
        return (y_vector,)
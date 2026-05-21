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
        # 1. Pre-asignar vector de ceros con el largo máximo en float32
        y_vector = np.zeros(self.max_actions, dtype=np.float32)

        n_actions = len(actions)
        if n_actions == 0:
            return (y_vector, )

        # 2. Definir el baseline de la heurística costosa g(x)
        # Como 'greedy' está ordenado por h(x), greedy[0] es el baseline del tope
        g_base = greedy[0]

        # 3. Calcular las ganancias netas (solo valores mayores a g_base)
        gains = np.array([max(0.0, g_val - g_base) for g_val in greedy[:n_actions]], dtype=np.float32)

        gains = np.power(gains, 2)
        sum_gains = np.sum(gains)

        # 4. Asignación del Target considerando todas las combinaciones
        if sum_gains > 0.0:
            # ESCENARIO 1: Al menos una acción mejoró el baseline.
            # El premio se va únicamente a las que superaron a g_base de forma proporcional.
            y_vector[:n_actions] = gains / sum_gains
            
        else:
            # ESCENARIO 2 y 3: Nadie superó el baseline.
            # Buscamos cuántas acciones empataron exactamente con el valor de g_base en el tope
            # Usamos np.isclose para evitar problemas de precisión milimétrica en flotantes
            tied_actions_mask = np.isclose(greedy[:n_actions], g_base)
            num_ties = np.sum(tied_actions_mask)
            
            # Se reparte el 1.0 equitativamente entre todos los que empataron en el primer lugar
            # Si no hay empates, num_ties será 1 y la acción 0 recibirá el 1.0 de forma natural.
            y_vector[:n_actions] = np.where(tied_actions_mask, 1.0 / num_ties, 0.0)

        return (y_vector, )
import numpy as np
from data.adapters.output.output_adapter import OutputAdapter
from data.objects import Action

class AdvantageAdapter(OutputAdapter):
    def __init__(self):
        super().__init__({
            "Y": np.float32
        })

    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: list):
        """
        Calcula el target escalar de ventaja relativa (volumen neto ganado)
        para la Red de Valor, manteniendo la firma original intacta.
        """
        # 1. Extraer los hitos del rollout directamente en volumen puro
        h_pure_vol = greedy[0]          # Volumen final alcanzado por h(x) analítica
        greedy_rollout_vol = max(greedy) # Máximo volumen alcanzado en la expansión

        # 2. Calcular la ventaja neta de volumen ganada en CPU
        # Si el rollout no mejora a h(x), esto da 0.0 perfecto.
        y_value = 100 * (greedy_rollout_vol - h_pure_vol)

        return (y_value, )
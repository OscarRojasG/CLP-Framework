import numpy as np

class Action:
    def __init__(self, data):
        self.block_id = int(data[0])
        self.vcs = data[1]
        self.loss = data[2]
        self.cs = data[3]

    def norm_vcs(self):
        # Si el valor es infinito, la exponencial de menos infinito tiende a 0,
        # por lo que 1 - 0 = 1.0 de forma directa y limpia.
        if np.isinf(self.vcs):
            return 1.0
            
        # Parámetro de escala calculado para que la media (0.0051) mapee cerca de 0.5
        gamma = 135.0
        
        # Mapea el rango [0, +inf] al rango acotado [0, 1]
        vcs_norm = 1.0 - np.exp(-gamma * self.vcs)
        
        return vcs_norm
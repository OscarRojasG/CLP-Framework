from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class Transformer(nn.Module, ABC):
    def __init__(self, seed=None, **hyperparams):
        # La semilla se fija antes de que la subclase cree sus capas, por lo que determina la inicialización.
        # No se guarda en hyperparams (no forma parte de la arquitectura).
        if seed is not None:
            torch.manual_seed(seed)
        super(Transformer, self).__init__()
        self.hyperparams = hyperparams
        self.biased = False
    
    @abstractmethod
    def encode(self, *args):
        pass

    @abstractmethod
    def decode(self, *args):
        pass

    def forward(self, block_data, *args):
        enc_data = self.encode(block_data)
        return self.decode(*enc_data, *args)
from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class Transformer(nn.Module, ABC):
    def __init__(self, **hyperparams):
        torch.manual_seed(42)
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
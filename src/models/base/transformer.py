from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class Transformer(nn.Module, ABC):
    def __init__(self, **hyperparams):
        torch.manual_seed(42)
        super(Transformer, self).__init__()
        self.hyperparams = hyperparams

    @abstractmethod
    def encode(self, block_features):
        pass

    @abstractmethod
    def decode(self, memory, *args, **kwargs):
        pass

    def forward(self, block_features, *decoder_args, **decoder_kwargs):
        memory = self.encode(block_features)
        return self.decode(memory, *decoder_args, **decoder_kwargs)
from abc import ABC, abstractmethod
import torch.nn as nn

class Transformer(nn.Module, ABC):
    def __init__(self, **hyperparams):
        super(Transformer, self).__init__()
        self.hyperparams = hyperparams

    @abstractmethod
    def encode(self, block_features):
        pass

    @abstractmethod
    def decode(self, memory, action_blocks, action_features, placed_blocks, placed_features, space_features):
        pass

    def forward(self, block_features, action_blocks, action_features, placed_blocks, placed_features, space_features):
        memory = self.encode(block_features)
        return self.decode(memory, action_blocks, action_features, placed_blocks, placed_features, space_features)
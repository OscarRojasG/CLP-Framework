import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """Clase base general para todos los modelos."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *inputs):
        """Método forward que debe implementarse en subclases."""
        pass

    @abstractmethod
    def predict(self, *inputs):
        """Devuelve predicciones a partir de los logits."""
        pass
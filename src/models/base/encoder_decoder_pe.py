import torch
from abc import ABC, abstractmethod
from .base import BaseModel

class EncoderDecoderPEModel(BaseModel):
    """
    Clase abstracta para modelos tipo encoder-decoder.
    Ejemplo: Transformer con codificador (X_src) y decodificador (X_tgt)
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, X_src, X_tgt):
        pass

    def predict(self, X_src, X_tgt, placed, coords, apply_softmax=False):
        logits = self.forward(X_src, X_tgt, placed, coords)
        if apply_softmax:
            return torch.softmax(logits, dim=-1)
        return logits
import torch
from abc import ABC, abstractmethod
from .base import BaseModel

class DecoderOnlyModel(BaseModel):
    """
    Clase abstracta para modelos que solo usan una secuencia (decoder-only).
    Ejemplo: DecoderModel
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, X_tgt):
        """
        Parámetros:
            X_tgt: Tensor [batch, seq_len, input_dim]
        Retorna:
            Logits o salidas del modelo
        """
        pass

    def predict(self, X_tgt, apply_softmax=False):
        logits = self.forward(X_tgt)
        if apply_softmax:
            return torch.softmax(logits, dim=-1)
        return logits
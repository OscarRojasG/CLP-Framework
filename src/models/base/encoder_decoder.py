import torch
from abc import ABC, abstractmethod
from .base import BaseModel

class EncoderDecoderModel(BaseModel):
    """
    Clase abstracta para modelos tipo encoder-decoder.
    Ejemplo: Transformer con codificador (X_src) y decodificador (X_tgt)
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, X_src, X_tgt):
        """
        Parámetros:
            X_src: Tensor [batch, src_len, src_dim]
            X_tgt: Tensor [batch, tgt_len, tgt_dim]
        Retorna:
            Logits o salidas del modelo
        """
        pass

    def predict(self, X_src, X_tgt, apply_softmax=False):
        logits = self.forward(X_src, X_tgt)
        if apply_softmax:
            return torch.softmax(logits, dim=-1)
        return logits
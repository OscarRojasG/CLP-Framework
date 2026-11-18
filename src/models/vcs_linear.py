import torch
import torch.nn as nn
from models.base.transformer import Transformer


class VCSLinear(Transformer):
    """
    Modelo lineal sobre los términos de VCS:
        s(b) = w_V log V(b) + w_CS log CS'(b,p) + w_L log(1 - L(b,r)) + w_N log N(b)

    Como log VCS = delta log V + alpha log CS' + beta log(1 - L) + gamma log N, este modelo es VCS
    con exponentes aprendidos (salvo un factor de escala, que el softmax absorbe; ver exponents()).

    Usa los mismos tensores que InputAdapterV1 (y que el TensorEncoder del motor C++), por lo que se
    entrena con los datasets actuales y se usa en BSG_Hybrid_Solver sin cambios:
        block_features[..., 3] = V(b) = volumen ocupado / volumen del contenedor
        block_features[..., 4] = N(b) = 1/n
        action_features[..., 0] = L(b,r) (recortada a 0 en los tensores; VCS usa la pérdida sin recortar, que puede ser negativa)
        action_features[..., 1] = CS'(b,p)
    placed_features se ignora (VCS no usa el contenedor).
    """
    FEATURES = ["V", "CS", "1-L", "N"]

    def __init__(self, block_dim=5, action_dim=2, placed_dim=6, seed=None):
        super().__init__(seed=seed, block_dim=block_dim, action_dim=action_dim, placed_dim=placed_dim)
        # Sin sesgo: un término constante por acción no cambia el softmax entre acciones
        self.linear = nn.Linear(len(self.FEATURES), 1, bias=False)
        self.eps = 1e-8

    def encode(self, block_features):
        # Las features de los bloques se usan directamente (no hay embedding)
        return block_features,

    def log_features(self, memory, action_blocks, action_features):
        action_mask = (action_blocks != -1)
        idx = action_blocks.clamp(min=0)
        blocks = torch.gather(memory, 1, idx.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))

        V, N = blocks[..., 3], blocks[..., 4]
        L, CS = action_features[..., 0], action_features[..., 1]
        feats = torch.stack([V, CS, 1 - L, N], dim=-1).clamp(min=self.eps).log()

        # El padding (-1) daría log de un número no positivo: se anula y luego se enmascara
        return feats.masked_fill(~action_mask.unsqueeze(-1), 0.0), action_mask

    def decode(self, memory, action_blocks, action_features, placed_features):
        feats, action_mask = self.log_features(memory, action_blocks, action_features)
        logits = self.linear(feats).squeeze(-1)
        return logits.masked_fill(~action_mask, -1e9)

    def weights(self):
        return dict(zip(self.FEATURES, self.linear.weight.detach().flatten().tolist()))

    def exponents(self):
        """Pesos normalizados por w_V (delta = 1, como en VCS) -> comparables con alpha, beta, gamma."""
        w = self.weights()
        return {"delta": 1.0, "alpha": w["CS"] / w["V"], "beta": w["1-L"] / w["V"], "gamma": w["N"] / w["V"]}

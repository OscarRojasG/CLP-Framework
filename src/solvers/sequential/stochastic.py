import torch
from models.base.transformer import Transformer
from solvers.sequential.base import BaseSolver


class StochasticModelSolver(BaseSolver):
    def __init__(self, model: Transformer, temperature: float = 1.0):
            super().__init__(model)
            self.temperature = temperature
            
    def solve(self, instance_file, instance_number, w: int, iters=1) -> float:
        best_vol = 0
        for _ in range(iters):
            vol = super().solve(instance_file, instance_number, w)
            best_vol = max(best_vol, vol)
        return best_vol
    
    def select_action(self, logits: torch.Tensor, action_blocks: torch.Tensor) -> int:
        # 1. Obtenemos probabilidades base (la temperatura aquí influye poco si luego igualamos)
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # 2. Creamos una máscara de ceros del mismo tamaño
        filtered_probs = torch.zeros_like(probs)
        
        # 3. Identificamos los índices de los 8 valores más altos
        # topk devuelve (values, indices)
        top_values, top_indices = torch.topk(probs, k=min(8, len(probs)))
        
        # 4. Seteamos esos índices a 0.125 (1/8)
        # Usamos 1.0 / len(top_indices) por si hay menos de 8 acciones disponibles
        val = 1.0 / len(top_indices)
        filtered_probs[top_indices] = val
        
        # 5. Muestreamos de la nueva distribución (ahora uniforme entre las top 8)
        idx = torch.multinomial(filtered_probs, num_samples=1).item()
        
        # --- Debug para testing ---
        sorted_indices = torch.argsort(probs, descending=True)
        rank = (sorted_indices == idx).nonzero(as_tuple=True)[0].item()
        print(f"[Test Top-8] Elegido Rank: {rank + 1}º | Prob original: {probs[idx]:.4f} -> Nueva: {val:.3f}")
        # --------------------------

        return int(action_blocks[idx])
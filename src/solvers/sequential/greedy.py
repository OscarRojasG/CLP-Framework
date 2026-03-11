import torch
from solvers.sequential.base import BaseSolver


class GreedyModelSolver(BaseSolver):
    def select_action(self, logits: torch.Tensor, action_blocks: torch.Tensor) -> int:
        idx = logits.argmax().item()
        return int(action_blocks[idx])
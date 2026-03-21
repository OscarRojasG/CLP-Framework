from solvers.bs.bsg import BSG_Solver
from solvers.greedy.greedy_solver import GreedySolver

class VCSSolver(GreedySolver):
    def __init__(self):
        super().__init__("VCS")
        self.bsg = BSG_Solver()

    # TODO: Crear ejecutable para VCS        
    def solve(self, instance_file, instance_number, min_fr) -> int:
        return self.bsg.solve(instance_file, instance_number, 1, min_fr)
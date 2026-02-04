from solvers.bsg import BSGSolver

class VCSSolver(BSGSolver):
    def solve(self, instance_file, instance_number) -> int:
        return super().solve(instance_file, instance_number, 1)
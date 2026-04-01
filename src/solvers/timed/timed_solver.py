from abc import ABC, abstractmethod
from solvers.dse.dse_solver import DSE_BSM_Solver

class Timed_Solver(ABC):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def solve(self, instance_file, instance_number, min_fr, time) -> int:
        pass
    
class Timed_BSM_Solver(Timed_Solver, DSE_BSM_Solver):
    def __init__(self, model, name, dse_module, solver_class):
        Timed_Solver.__init__(self, name)
        DSE_BSM_Solver.__init__(self, model, name, dse_module, solver_class)
        self.verbose = False
        
    def solve(self, instance_file, instance_number, min_fr, time):
        return self._solve(instance_file, instance_number, min_fr, time, 999999)[0]
from abc import ABC, abstractmethod
from settings import INSTANCE_FOLDER

class Timed_Solver(ABC):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def solve(self, instance_file, instance_number, min_fr, time) -> int:
        pass
    
class Timed_BSM_Solver(Timed_Solver):
    def __init__(self, model, name, dse_module, solver_class):
        super().__init__(name)
        self.model = model
        self.dse_module = dse_module
        self.solver_class = solver_class
        self.verbose = False
        
    def solve(self, instance_file, instance_number, min_fr, time):
        instance_file = str(INSTANCE_FOLDER / instance_file)       
        dse = self.dse_module(instance_file, instance_number, min_fr, time)
        solver = self.solver_class(self.model)
        best_volume = 0
        
        while not dse.is_finished():
            if self.verbose:
                print(f"Ejecutando {self.name} con w =", dse.w)
                
            env = dse.get_env()
            solver.solve_from_env(env)
            dse.update()
            
            volume = dse.best_volume * 100
            if volume > best_volume:
                best_volume = volume
                if self.verbose:
                    print("Actualizando mejor volumen:", volume)
            
        return dse.best_volume * 100
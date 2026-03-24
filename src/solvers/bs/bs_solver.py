from abc import ABC, abstractmethod
from settings import INSTANCE_FOLDER
import os

class BS_Solver(ABC):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def solve(self, instance_file, instance_number, w, min_fr) -> int:
        pass
    
class BSM_Solver(BS_Solver):
    def __init__(self, model, name, solver_module):
        super().__init__(name)
        self.model = model
        self.solver_module = solver_module
        
    def load_env(self, instance_file, instance_number, w, min_fr):
        instance_file = str(INSTANCE_FOLDER / instance_file) 
        
        if os.path.exists(instance_file) == False:
            raise Exception(f'El archivo de instancia {instance_file} no existe.')
        
        return self.solver_module(instance_file, instance_number, w, min_fr)
    
    def solve(self, instance_file, instance_number, w, min_fr):
        bsm = self.load_env(instance_file, instance_number, w, min_fr)
        vol, time = self.solve_from_env(bsm)
        del bsm
        return vol, time
    
    @abstractmethod
    def solve_from_env(self, bsm):
        pass
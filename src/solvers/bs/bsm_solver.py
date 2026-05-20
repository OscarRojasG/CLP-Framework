from abc import abstractmethod
from settings import INSTANCE_FOLDER
import os
from solvers.solver import Solver
    
class BSMSolver(Solver):
    def __init__(self, name, solver_module, w, min_fr):
        super().__init__(name, min_fr)
        self.solver_module = solver_module
        self.w = w
        
    def load_env(self, file, instance_number):
        file = str(INSTANCE_FOLDER / file) 
        
        if os.path.exists(file) == False:
            raise Exception(f'El archivo de instancia {file} no existe.')
        
        return self.solver_module(file, instance_number, self.w, self.min_fr)
    
    def solve(self, file, instance_number):
        env = self.load_env(file, instance_number)
        vol, time = self.solve_from_env(env)
        del env
        return vol, time
    
    @abstractmethod
    def solve_from_env(self, env):
        pass
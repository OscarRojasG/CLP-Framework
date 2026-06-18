from abc import abstractmethod
from settings import INSTANCE_FOLDER
import os
from solvers.solver import Solver
    
class BSMSolver(Solver):
    def __init__(self, name, solver_module, w, min_fr, input_adapter, inference_mode):
        super().__init__(name, min_fr)
        self.solver_module = solver_module
        self.w = w
        self.input_adapter = input_adapter
        self.inference_mode = inference_mode
        
    def load_env(self, file, instance_number):
        file = str(INSTANCE_FOLDER / file) 
        
        if not os.path.exists(file):
            raise Exception(f'El archivo de instancia {file} no existe.')
        
        # Si tenemos adapter y estamos en inferencia, pasamos los parámetros extra
        if self.inference_mode and self.input_adapter:
            return self.solver_module(
                file, 
                instance_number, 
                self.w, 
                self.input_adapter.max_blocks,
                self.input_adapter.max_actions,
                self.input_adapter.max_pblocks,
                self.min_fr
            )
        else:
            return self.solver_module(file, instance_number, self.w, self.min_fr)
    
    def solve(self, file, instance_number):
        env = self.load_env(file, instance_number)
        vol, time = self.solve_from_env(env)
        del env
        return vol, time
    
    @abstractmethod
    def solve_from_env(self, env):
        pass
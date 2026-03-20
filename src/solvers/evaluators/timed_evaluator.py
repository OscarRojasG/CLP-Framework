from solvers.timed.timed_solver import Timed_Solver
import pandas as pd

def timed_eval(solver_list: list[Timed_Solver], instance_file: str, num_instances: int, min_fr: float, time: float, start: int = 0):
    eval_dict = {
        'instance': [i for i in range(start, start + num_instances)],
    }
    
    for solver in solver_list:
        vols = []
        
        for i in range(start, start + num_instances):
            vol = solver.solve(instance_file, i, min_fr, time)
            vols.append(vol)
            
        eval_dict[solver.name] = vols
        
    return pd.DataFrame(eval_dict)
from solvers.greedy.greedy_solver import GreedySolver
import pandas as pd
import time

def greedy_eval(solver_list: list[GreedySolver], instance_file: str, num_instances: int, min_fr: float):
    eval_dict = {
        'instance': [i for i in range(num_instances)],
    }
    
    for solver in solver_list:
        vols = []
        times = []
        
        for i in range(num_instances):
            vol, time = solver.solve(instance_file, i, min_fr)
            vols.append(vol)
            times.append(time)
            
        eval_dict['Vol ' + solver.name] = vols
        eval_dict['Time ' + solver.name] = times
        
    return pd.DataFrame(eval_dict)
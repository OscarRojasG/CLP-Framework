from solvers.bs.bs_solver import BS_Solver
import pandas as pd
import time

def bs_eval(solver_list: list[BS_Solver], instance_file: str, num_instances: int, w: int):
    eval_dict = {
        'instance': [i for i in range(num_instances)],
    }
    
    for solver in solver_list:
        vols = []
        times = []
        
        for i in range(num_instances):
            t0 = time.perf_counter()
            vol = solver.solve(instance_file, i, w)
            t1 = time.perf_counter()
            vols.append(vol)
            times.append(t1 - t0)
            
        eval_dict['Vol ' + solver.name] = vols
        eval_dict['Time ' + solver.name] = times
        
    return pd.DataFrame(eval_dict)
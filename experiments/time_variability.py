import torch
import gc
import statistics
import pandas as pd
from solvers.timed.timed_bsm_vcs import Timed_BSM_VCS_Solver
from models.CLPTransformer_v4 import CLPTransformer
from training import load_model

model_name = "CLPTransformer_v4_CL"
instance_file = "benchmarks/BR4.txt"
instance_number = 1
repetitions = 20
time = 6

def run():
    model = load_model(CLPTransformer, model_name)
    solver = Timed_BSM_VCS_Solver(model)
    results = []
    
    print("Iniciando experimento")
    
    for i in range(repetitions):
        gc.collect() # Limpieza profunda
        torch.cuda.empty_cache()
        
        val = solver.solve(instance_file, instance_number, time)
        results.append(val)
        
        pd.DataFrame(results).to_csv("time_variability.csv", index=False)
        print(f"Iteración {i+1}/{repetitions} completada.")
        
    mean_val = statistics.mean(results)
    std_dev = statistics.stdev(results)

    print(mean_val)
    print(std_dev)

if __name__ == "__main__":
    run()
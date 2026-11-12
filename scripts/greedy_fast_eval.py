from models.clp_model import CLPModel
from solvers.evaluator import fast_eval
from solvers.greedy import GreedyModelSolver
from data.adapters.input.v1 import InputAdapterV1

model_name = "M30"

w = 8
num_instances = 100
adapter_config = (InputAdapterV1, 10000, 64, w*w)

for k in range(1, 16):
    instance_file = f"benchmarks/BR{k}.txt"
    output_csv = f"greedy_fast_{model_name}_BR{k}.csv"
    min_fr = 1 if k <= 7 else 0.98

    solvers_config = [
        (GreedyModelSolver, w, min_fr, True)
    ]

    fast_eval(solvers_config, instance_file, num_instances, CLPModel, model_name, adapter_config, output_csv=output_csv)

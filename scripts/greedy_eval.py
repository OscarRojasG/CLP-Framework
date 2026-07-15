from models.clp_transformer import CLPTransformer
from solvers.evaluator import fast_eval
from solvers.greedy import GreedyModelSolver, VCSSolver
from data.adapters.input.v1 import InputAdapterV1

instance_file = "benchmarks/BR8.txt"
output_csv = "greedy_eval_BR8.csv"
model_name = "M30"
w = 8
min_fr = 0.98

solvers_config = [
    (GreedyModelSolver, w, min_fr),
    (VCSSolver, min_fr)
]
num_instances = 100
adapter_config = (InputAdapterV1, 10000, 1, w*w)

fast_eval(solvers_config, instance_file, num_instances, CLPTransformer, model_name, adapter_config, output_csv=output_csv)
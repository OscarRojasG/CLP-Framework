from models.clp_model import CLPModel
from training.training import load_model
from solvers.evaluator import run_eval
from solvers.greedy import GreedyModelSolver
from data.adapters.input.v1 import InputAdapterV1

model_name = "M30"

w = 8
num_instances = 100
output_dir = "greedy"  # relativo a EXPERIMENTS_FOLDER

model = load_model(CLPModel, model_name)
input_adapter = InputAdapterV1(max_blocks=10000, max_pblocks=1, max_actions=w*w)

for k in range(1, 16):
    instance_file = f"benchmarks/BR{k}.txt"
    output_csv = f"{output_dir}/greedy_{model_name}_BR{k}.csv"
    min_fr = 1 if k <= 7 else 0.98

    solver = GreedyModelSolver(model, input_adapter, w, min_fr)

    run_eval([solver], instance_file, num_instances, output_csv=output_csv)

from models.clp_transformer import CLPTransformer
from training.training import load_model
from solvers.evaluator import run_eval
from solvers.beam_search import BSG_ModelVCS_Solver
from data.adapters.input.v1 import InputAdapterV1

instance_file = "benchmarks/BR8.txt"
output_csv = "model_vcs_eval_BR8.csv"
model_name = "M30"
w = 8
min_fr = 0.98

model = load_model(CLPTransformer, model_name)
input_adapter = InputAdapterV1(max_blocks=10000, max_pblocks=64, max_actions=w*w)
solver = BSG_ModelVCS_Solver(model, input_adapter, w, min_fr)
num_instances = 100

run_eval([solver], instance_file, num_instances, output_csv=output_csv)
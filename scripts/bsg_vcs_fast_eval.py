from solvers.evaluator import fast_eval
from solvers.beam_search import BSG_VCS_Solver

w = 8
num_instances = 100

for k in range(1, 16):
    instance_file = f"benchmarks/BR{k}.txt"
    output_csv = f"bsg_vcs_fast_BR{k}.csv"
    min_fr = 1 if k <= 7 else 0.98

    solvers_config = [
        (BSG_VCS_Solver, w, min_fr)
    ]

    fast_eval(solvers_config, instance_file, num_instances, output_csv=output_csv)

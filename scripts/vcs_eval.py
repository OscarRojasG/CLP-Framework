from solvers.evaluator import run_eval
from solvers.greedy import VCSSolver

num_instances = 100
output_dir = "vcs"  # relativo a EXPERIMENTS_FOLDER

for k in range(1, 16):
    instance_file = f"benchmarks/BR{k}.txt"
    output_csv = f"{output_dir}/vcs_BR{k}.csv"
    min_fr = 1 if k <= 7 else 0.98

    solver = VCSSolver(min_fr)

    run_eval([solver], instance_file, num_instances, output_csv=output_csv)

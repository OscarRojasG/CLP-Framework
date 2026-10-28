from solvers.evaluator import run_eval
from solvers.beam_search import BSG_VCS_Solver

w = 8
num_instances = 100
benchmarks = range(1, 16)  # BR1..BR15; también sirve una lista, p. ej. [8, 9, 10]
output_dir = "bsg_vcs"  # relativo a EXPERIMENTS_FOLDER

for k in benchmarks:
    instance_file = f"benchmarks/BR{k}.txt"
    output_csv = f"{output_dir}/bsg_vcs_w{w}_BR{k}.csv"
    min_fr = 1 if k <= 7 else 0.98

    solver = BSG_VCS_Solver(w, min_fr)

    run_eval([solver], instance_file, num_instances, output_csv=output_csv)

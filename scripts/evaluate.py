"""
Evaluation of the methods in the paper on the BR1-BR15 (test) or V1-V15 (validation) benchmarks.

Examples (each one produces the files read by the notebooks):
    python evaluate.py vcs                                          # VCS
    python evaluate.py greedy --model MIX_MEDIAN                    # Greedy (median model)
    python evaluate.py bsg-vcs --w 16                               # BSG-VCS, w=16
    python evaluate.py bsg-hybrid --model MIX_MEDIAN --w 16         # BSG-Hybrid, w=16
    python evaluate.py de-bsg-vcs --t 150                           # BSG-VCS under a time limit
    python evaluate.py de-bsg-hybrid --model MIX_MEDIAN --t 150     # BSG-Hybrid under a time limit
    python evaluate.py bsg-hybrid --model MIX_ENC4_S42 --fast       # comparison of models and seeds
    python evaluate.py bsg-hybrid --model MIX_ENC4_S42 --fast --bench V   # validation
    python evaluate.py bsg-vcs --fast --fixed-w                     # expansion rule ablation

By default the evaluation is sequential (single thread), with computation times comparable across methods,
and the results are stored in experiments/<method>/. With --fast, instances are solved in parallel
(times are not comparable) and the results are stored in experiments/fast/.
"""
import argparse
from data.adapters.input.v1 import InputAdapterV1
from models.clp_model import CLPModel
from models.clp_model_no_cont import CLPModelNoCont
from models.vcs_linear import VCSLinear
from solvers.beam_search import BSG_VCS_Solver, BSG_Hybrid_Solver
from solvers.double_effort import DE_BSG_VCS_Solver, DE_BSG_Hybrid_Solver
from solvers.evaluator import run_eval, fast_eval
from solvers.env_solver import EnvSolver
from solvers.greedy import VCSSolver, GreedyModelSolver
from training.training import load_hyperparams, load_model

METHODS = ["vcs", "greedy", "bsg-vcs", "bsg-hybrid", "de-bsg-vcs", "de-bsg-hybrid"]
MODEL_METHODS = {"greedy", "bsg-hybrid", "de-bsg-hybrid"}
BENCHMARKS = {"BR": "benchmarks/BR{k}.txt", "V": "val/V{k}.txt"}


def model_class(model_name):
    """Model class according to its saved hyperparameters."""
    hyperparams = load_hyperparams(model_name)
    if "d_model" not in hyperparams:
        return VCSLinear
    if "cont_layers" not in hyperparams:
        return CLPModelNoCont
    return CLPModel


def solver_config(args, min_fr):
    """(class, *arguments) of the solver. Solvers that use a model also receive the model and the input adapter."""
    if args.method == "vcs":
        return (VCSSolver, min_fr)
    if args.method == "greedy":
        return (GreedyModelSolver, args.w, min_fr)
    if args.method == "bsg-vcs":
        return (BSG_VCS_Solver, args.w, min_fr, args.fixed_w)
    if args.method == "bsg-hybrid":
        return (BSG_Hybrid_Solver, args.w, min_fr, args.fixed_w)
    if args.method == "de-bsg-vcs":
        return (DE_BSG_VCS_Solver, args.t, min_fr)
    return (DE_BSG_Hybrid_Solver, args.t, min_fr)


def output_name(args, set_name):
    """Name of the results CSV (relative to experiments/)."""
    name = args.method.replace("-", "_")
    if args.method in MODEL_METHODS:
        model = f"_{args.model}"
    else:
        model = ""

    # Expansion rule: only indicated when it is not the method's own (dynamic in BSG-VCS, fixed in BSG-Hybrid)
    expansion = ""
    if args.method == "bsg-vcs" and args.fixed_w:
        expansion = "_fixedw"
    elif args.method == "bsg-hybrid" and not args.fixed_w:
        expansion = "_dynw"

    if args.fast:
        width = f"_w{args.w}" if args.method in ("greedy", "bsg-vcs", "bsg-hybrid") and args.w != 8 else ""
        limit = f"_t{args.t}" if args.method.startswith("de-") else ""
        return f"fast/{name}{expansion}_fast{model}{width}{limit}_{set_name}.csv"

    width = f"_w{args.w}" if args.method in ("bsg-vcs", "bsg-hybrid") else ""
    limit = f"_t{args.t}" if args.method.startswith("de-") else ""
    return f"{name}/{name}{expansion}{model}{width}{limit}_{set_name}.csv"


def main():
    parser = argparse.ArgumentParser(description="Evaluates a method on BR1-BR15 or V1-V15.")
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--model", default="MIX_MEDIAN", help="model (for greedy, bsg-hybrid and de-bsg-hybrid)")
    parser.add_argument("--w", type=int, default=8, help="beam width")
    parser.add_argument("--t", type=int, default=30, help="time limit in seconds (de-bsg-*)")
    parser.add_argument("--bench", choices=list(BENCHMARKS), default="BR")
    parser.add_argument("--sets", type=int, nargs="+", default=list(range(1, 16)), help="sets to evaluate (1-15)")
    parser.add_argument("--instances", type=int, default=100, help="instances per set")
    parser.add_argument("--fast", action="store_true", help="solves the instances in parallel")
    expansion = parser.add_mutually_exclusive_group()
    expansion.add_argument("--fixed-w", dest="fixed_w", action="store_true", default=None,
                           help="fixed expansion rule (default in bsg-hybrid)")
    expansion.add_argument("--dynamic-w", dest="fixed_w", action="store_false",
                           help="dynamic expansion rule (default in bsg-vcs)")
    args = parser.parse_args()
    if args.fixed_w is None:
        args.fixed_w = args.method == "bsg-hybrid"

    uses_model = args.method in MODEL_METHODS
    model_cls = model_class(args.model) if uses_model else None
    adapter_config = (InputAdapterV1, 10000, 64, args.w * args.w)

    if not args.fast and uses_model:
        model = load_model(model_cls, args.model)
        adapter_cls, *adapter_args = adapter_config
        input_adapter = adapter_cls(*adapter_args)

    for k in args.sets:
        instance_file = BENCHMARKS[args.bench].format(k=k)
        set_name = f"{args.bench}{k}"
        # Sets 1-7 are weakly heterogeneous, 8-15 strongly heterogeneous
        min_fr = 1 if k <= 7 else 0.98
        config = solver_config(args, min_fr)
        output_csv = output_name(args, set_name)

        if args.fast:
            fast_eval([config], instance_file, args.instances, model_cls, args.model, adapter_config,
                      output_csv=output_csv)
        else:
            solver_cls, *solver_args = config
            if issubclass(solver_cls, EnvSolver):
                solver = solver_cls(model, input_adapter, *solver_args)
            else:
                solver = solver_cls(*solver_args)
            run_eval([solver], instance_file, args.instances, output_csv=output_csv)


if __name__ == "__main__":
    main()

"""
Generation of the training data of an instance set (M3, M30 or M100).

    python generate_data.py M3
    python generate_data.py M30
    python generate_data.py M100

For each instance file of the set:
  1. Generates the instances (instances/<set>-<seed>.txt).
  2. Solves each instance with BSG-VCS (w=64) and stores the trace of the best solution (outputs/<set>-<seed>/).
  3. Extracts the state-action pairs and applies the filters (data/actions_<set>-<seed>.data).
Finally, the samples of all files are split into three difficulty levels according to the rank of the target
action under VCS, and 12,000 samples are drawn at random from each level (data/datasets/{easy,medium,hard}_<set>.data).
"""
import argparse
import os
from data.adapters.input.v1 import InputAdapterV1
from data.adapters.output.action_adapter import ActionAdapter
from data.generation import run_instances_parallel, generate_data
from data.preprocessing import generate_datasets
from instances.generation import generate_instances
from settings import DATASETS_FOLDER

# Set -> (Bischoff and Ratcliff configuration group, box types, files, instances per file)
SETS = {
    "M3": (1, 3, 8, 1000),
    "M30": (8, 30, 7, 500),
    "M100": (15, 100, 9, 200),
}
FIRST_SEED = 42

# Search that generates the demonstrations
W = 64
NUM_ACTIONS = 64

# Filters
MIN_BLOCKS = 100
MIN_ACTIONS = 4
MAX_ACTIONS = 64

# Difficulty levels according to the rank of the target action under VCS: easy [1, 1], medium [2, 8], hard [9, 64]
CUTS = [1, 1, 8, 64]
LEVELS = ["easy", "medium", "hard"]
SAMPLES_PER_LEVEL = 12000


def main():
    parser = argparse.ArgumentParser(description="Generates the training data of an instance set.")
    parser.add_argument("set", choices=list(SETS))
    parser.add_argument("--workers", type=int, default=None, help="parallel processes for BSG-VCS")
    args = parser.parse_args()

    group, n_types, n_files, n_instances = SETS[args.set]
    # Sets 1-7 are weakly heterogeneous, 8-15 strongly heterogeneous
    min_fr = 1 if group <= 7 else 0.98

    configs = []
    for seed in range(FIRST_SEED, FIRST_SEED + n_files):
        config = f"{args.set}-{seed}"
        configs.append(config)
        print(f"\n─── {config} ───")

        # The generator seed concatenates the group and the file seed (e.g. group 8, seed 42 -> 842)
        generate_instances(f"{config}.txt", n_instances, n_types, int(f"{group}{seed}"))
        run_instances_parallel(f"{config}.txt", W, NUM_ACTIONS, min_fr, max_workers=args.workers)
        generate_data(config, InputAdapterV1(max_blocks=10000, max_pblocks=100, max_actions=MAX_ACTIONS),
                      ActionAdapter(max_actions=MAX_ACTIONS), MIN_BLOCKS, MIN_ACTIONS, MAX_ACTIONS, prefix="actions")

    # Datasets per difficulty level (generate_datasets stores them as <set>_<start>-<end>.data)
    generate_datasets([f"actions_{config}.data" for config in configs], args.set, CUTS, max_size=SAMPLES_PER_LEVEL)
    for level, (start, end) in zip(LEVELS, [(1, 1), (2, 8), (9, 64)]):
        os.replace(DATASETS_FOLDER / f"{args.set}_{start}-{end}.data", DATASETS_FOLDER / f"{level}_{args.set}.data")


if __name__ == "__main__":
    main()

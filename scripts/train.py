"""
Training of the models in the paper.

Examples:
    python train.py MIX_ENC4_S42 --data MIX --seed 42                     # M-Mix (proposed model)
    python train.py M30_S43 --data M30 --seed 43                          # M30
    python train.py MIX_WBAL_S44 --data MIX --weights balanced --seed 44  # loss weighting analysis
    python train.py MIX_NOCONT_S42 --data MIX --model nocont --seed 42    # ablation without the container stream
    python train.py LINEAR_S42 --data MIX --model linear --seed 42        # linear model over the VCS terms

Datasets are read from data/datasets/{easy,medium,hard}_{M3,M30,M100}.data.
The model is saved to models/<name>.pth (hyperparameters in models/hyperparams/<name>.json)
and the training history to models/logs/<name>.json.
"""
import argparse
import shutil
from data.preprocessing import load_dataset
from models.clp_model import CLPModel
from models.clp_model_no_cont import CLPModelNoCont
from models.vcs_linear import VCSLinear
from training.training import train, save_model, LRConfig
from training.metrics import CrossEntropyLoss, Accuracy
from settings import EXPERIMENTS_FOLDER, MODELS_FOLDER

INSTANCE_SETS = ["M3", "M30", "M100"]
LEVELS = ["easy", "medium", "hard"]

# Loss weight of each difficulty level (easy, medium, hard)
WEIGHTS = {
    "hard": [0.25, 0.25, 0.50],
    "medium": [0.25, 0.50, 0.25],
    "easy": [0.50, 0.25, 0.25],
    "balanced": [1 / 3, 1 / 3, 1 / 3],
}

# Architectural hyperparameters (hyperparameter table of the paper)
ARCHITECTURE = dict(block_dim=5, action_dim=2, placed_dim=6, d_model=64, nhead=2, enc_layers=4,
                    cont_layers=2, scoring_layers=2, ff_dim_multiplier=2, dropout=0.1)

# Training configuration
EPOCHS = 300
TRAIN_SIZE, TEST_SIZE = 9000, 3000  # per difficulty level; in M-Mix they are split among the three sets
BATCH_SIZE = 32
LR = LRConfig(start=1e-4, factor=0.5, patience=10, min=1e-5)
WEIGHT_DECAY = 1e-5
PATIENCE = 20


def build_model(kind, seed):
    if kind == "clp":
        return CLPModel(**ARCHITECTURE, seed=seed)
    if kind == "nocont":
        return CLPModelNoCont(**{k: v for k, v in ARCHITECTURE.items() if k != "cont_layers"}, seed=seed)
    return VCSLinear(seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Trains a model of the paper.")
    parser.add_argument("name", help="name under which the model is saved")
    parser.add_argument("--data", choices=INSTANCE_SETS + ["MIX"], default="MIX")
    parser.add_argument("--model", choices=["clp", "nocont", "linear"], default="clp")
    parser.add_argument("--weights", choices=list(WEIGHTS), default="hard")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sets = INSTANCE_SETS if args.data == "MIX" else [args.data]
    datasets = [load_dataset(f"{level}_{s}.data") for s in sets for level in LEVELS]
    # In M-Mix each weight is divided among the three sets, like the samples of each level
    loss_weights = [w / len(sets) for _ in sets for w in WEIGHTS[args.weights]]
    train_size, test_size = TRAIN_SIZE // len(sets), TEST_SIZE // len(sets)

    model = build_model(args.model, args.seed)
    model = train(model, EPOCHS, datasets, train_size, test_size, BATCH_SIZE, LR, WEIGHT_DECAY,
                  CrossEntropyLoss(), loss_weights, PATIENCE, [Accuracy()], args.seed)

    save_model(model, args.name)

    # train() writes the history to experiments/experiment_logs.json; it is stored next to the model
    (MODELS_FOLDER / "logs").mkdir(parents=True, exist_ok=True)
    shutil.move(EXPERIMENTS_FOLDER / "experiment_logs.json", MODELS_FOLDER / "logs" / f"{args.name}.json")


if __name__ == "__main__":
    main()

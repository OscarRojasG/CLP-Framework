# CLP-Framework

Code, trained models and experimental results for the paper

> **Learning to Refine Heuristic-Guided Decisions for the Container Loading Problem**
> Oscar Rojas and Ignacio Araya.

The framework trains a neural model to rank the candidate blocks of the Container Loading Problem (CLP) from demonstrations of BSG-VCS, the beam search of [Araya, Guerrero and Nuñez (2017)](https://doi.org/10.1016/j.cor.2017.01.002), and uses it inside the beam search (BSG-Hybrid). The search itself (block generation, state transitions and VCS rollouts) runs in C++ ([Metasolver](https://github.com/OscarRojasG/Metasolver)); this repository contains the model, the training pipeline and the evaluation code.

## Methods

| Method | Description |
|---|---|
| **VCS** | Greedy construction guided by the VCS heuristic. |
| **Greedy** | Greedy construction guided by the proposed model. |
| **BSG-VCS** | Beam search with VCS for both the expansion and the rollouts (dynamic expansion rule). |
| **BSG-Hybrid** | Beam search in which the proposed model ranks the candidates during the expansion, while rollouts are performed with VCS (fixed expansion rule). |

Under a time limit, both beam search methods use the iterative scheme of BSG-VCS (*double effort*), which restarts the search with a beam width increased by a factor of √2 until the time limit is reached.

## Repository structure

```
bin/            C++ executables (compiled from Metasolver, see Installation)
data/datasets/  Training datasets (not included, see Training)
experiments/    Results of all the experiments reported in the paper (one CSV per method and instance set)
instances/      BR benchmark (benchmarks/) and validation benchmark (val/)
models/         Trained models (.pth) and their hyperparameters (hyperparams/*.json)
notebooks/      Notebooks that reproduce the tables of the paper from experiments/
scripts/        Data generation, training and evaluation scripts
src/            Python package: models, solvers, training and data pipeline
```

## Installation

The framework has been tested on Linux with Python 3.14.

**1. Clone the repository and install the Python dependencies**

```bash
git clone https://github.com/OscarRojasG/CLP-Framework.git
cd CLP-Framework
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the `src/` package in editable mode, so its modules (`models`, `solvers`, `settings`, ...) can be imported from the scripts and notebooks.

**2. Compile the C++ components**

The executables and the Python engine are compiled from [Metasolver](https://github.com/OscarRojasG/Metasolver). Since the engine is a Python extension, configure the build from the same virtual environment so that it is compiled for the same Python interpreter:

```bash
git clone https://github.com/OscarRojasG/Metasolver.git
cd Metasolver
git submodule update --init --recursive
cmake . -DPython_EXECUTABLE=$(which python)
make BSG_VCS BSG_VCS_FIXED_W VCS clp_engine
```

**3. Copy the compiled files into the framework**

```bash
mkdir -p ../CLP-Framework/bin
cp BSG_VCS BSG_VCS_FIXED_W VCS ../CLP-Framework/bin/
cp clp_engine.*.so ../CLP-Framework/src/envs/
```

| File | Used by |
|---|---|
| `bin/BSG_VCS` | BSG-VCS and the generation of the training data |
| `bin/BSG_VCS_FIXED_W` | BSG-VCS with the fixed expansion rule (ablation) |
| `bin/VCS` | VCS |
| `src/envs/clp_engine.*.so` | Greedy and BSG-Hybrid (search engine driven from Python) |

## Reproducing the results

### Tables

The results of all the experiments in the paper are included in `experiments/`, so the tables can be reproduced directly with the notebooks:

| Notebook | Contents |
|---|---|
| `notebooks/validation.ipynb` | Validation benchmark (V1–V15): hyperparameter selection, median seed selection, model comparison and loss weighting analysis. |
| `notebooks/test.ipynb` | BR benchmark (BR1–BR15): model comparison, performance evaluation, scalability, time limits, loss weighting analysis and ablation studies. |

### Evaluation

`scripts/evaluate.py` evaluates a method on the 15 sets of the BR benchmark (`--bench BR`, default) or of the validation benchmark (`--bench V`), and writes the results to `experiments/` with the names read by the notebooks. Run the scripts from the `scripts/` folder:

```bash
cd scripts
python evaluate.py <method> [--model MODEL] [--w W] [--t T] [--bench {BR,V}] [--fast] [--fixed-w | --dynamic-w]
```

By default, instances are solved sequentially on a single thread, so that computation times are comparable across methods; these are the runs used for the computation times reported in the paper. With `--fast`, instances are solved in parallel, which is much faster but makes the measured times meaningless; it is used for the experiments that only report volume utilization. Sequential results are stored in `experiments/<method>/` and parallel results in `experiments/fast/`.

| Experiment | Commands |
|---|---|
| Validation (hyperparameter selection, median seed, model comparison, loss weighting) | `python evaluate.py bsg-hybrid --fast --bench V --model <model>_S<seed>`, with the reference `python evaluate.py bsg-vcs --fast --bench V` |
| Model comparison | `python evaluate.py greedy --fast --model <model>_S<seed>`<br>`python evaluate.py bsg-hybrid --fast --model <model>_S<seed>`<br>for `<model>` ∈ {`M3`, `M30`, `M100`, `MIX_ENC4`} and `<seed>` ∈ {42, 43, 44}, with the references `python evaluate.py vcs` and `python evaluate.py bsg-vcs --w 8` |
| Performance evaluation | `python evaluate.py vcs`<br>`python evaluate.py greedy --model MIX_MEDIAN`<br>`python evaluate.py bsg-vcs --w <w>`<br>`python evaluate.py bsg-hybrid --model MIX_MEDIAN --w <w>`<br>for `<w>` ∈ {8, 16} |
| Scalability | the last two commands above for `<w>` ∈ {2, 4, 8, 16, 32} |
| Time limits | `python evaluate.py de-bsg-vcs --t <t>`<br>`python evaluate.py de-bsg-hybrid --model MIX_MEDIAN --t <t>`<br>for `<t>` ∈ {30, 150} |
| Loss weighting analysis | `python evaluate.py bsg-hybrid --fast --model <model>_S<seed>` for `<model>` ∈ {`MIX_ENC4`, `MIX_WMEDIUM`, `MIX_WEASY`, `MIX_WBAL`} |
| Ablation: container stream | `python evaluate.py bsg-hybrid --fast --model MIX_NOCONT_S<seed>` |
| Ablation: scoring function and expansion rule | `python evaluate.py bsg-vcs --fast` and `python evaluate.py bsg-vcs --fast --fixed-w`<br>`python evaluate.py bsg-hybrid --fast --model <model>` and `python evaluate.py bsg-hybrid --fast --model <model> --dynamic-w`<br>for `<model>` ∈ {`LINEAR_MEDIAN`, `MIX_MEDIAN`} |

Run `python evaluate.py --help` for all the options (e.g. `--sets 8 9 10` to evaluate only some sets).

### Training

The training datasets used in the paper are available on Zenodo: [https://doi.org/10.5281/zenodo.24325230](https://doi.org/10.5281/zenodo.24325230). To train the models without regenerating the data, download the nine files into `data/datasets/`.

The same data can be regenerated by solving randomly generated instances with BSG-VCS (w=64) and recording its decisions:

```bash
python generate_data.py M3      # 8 files x 1,000 instances with 3 box types
python generate_data.py M30     # 7 files x 500 instances with 30 box types
python generate_data.py M100    # 9 files x 200 instances with 100 box types
```

This produces `data/datasets/{easy,medium,hard}_{M3,M30,M100}.data`, with 12,000 samples per difficulty level. `python analyze_discards.py <set>` reports the data generation statistics (discarded samples and difficulty split) of a set. Generating the data takes several hours, since every instance is solved with a wide beam.

A model is then trained with `train.py`, which takes the name under which the model is saved:

```bash
python train.py MIX_ENC4_S42 --data MIX --seed 42                     # M-Mix (proposed model)
python train.py M30_S42 --data M30 --seed 42                          # M30 (likewise M3 and M100)
python train.py MIX_WBAL_S42 --data MIX --weights balanced --seed 42  # loss weighting analysis
python train.py MIX_NOCONT_S42 --data MIX --model nocont --seed 42    # ablation without the container stream
python train.py LINEAR_S42 --data MIX --model linear --seed 42        # linear model over the VCS terms
```

The model is saved to `models/`, where it can be used by `evaluate.py` through `--model`.

## Models

All the models reported in the paper are included in `models/`, each trained with seeds 42, 43 and 44:

| Name | Description |
|---|---|
| `M3_S*`, `M30_S*`, `M100_S*` | Models trained on a single instance set (M3, M30, M100). |
| `MIX_ENC4_S*` | M-Mix, the proposed model, trained on the three instance sets. `ENC4` is the configuration selected in the hyperparameter search. |
| `MIX_MEDIAN` | Copy of the median seed of M-Mix (`MIX_ENC4_S42`), used in the performance evaluation. |
| `MIX_<variant>_S*` | Configurations of the hyperparameter search (`DIM32`, `DIM64`, `DIM128`, `ENC1`, `CONT1`, `CONT4`, `SCORING1`, `SCORING4`, `NHEAD1`, `NHEAD4`, `DROPOUT00`, `DROPOUT02`). |
| `MIX_WMEDIUM_S*`, `MIX_WEASY_S*`, `MIX_WBAL_S*` | M-Mix with the Medium-biased, Easy-biased and Balanced loss weighting schemes. |
| `MIX_NOCONT_S*` | M-Mix without the container stream (ablation). |
| `LINEAR_S*`, `LINEAR_MEDIAN` | Linear model over the VCS terms (ablation), and a copy of its median seed (`LINEAR_S43`). |

The architecture of each model is inferred from its hyperparameters when it is loaded with `evaluate.py`.

## License

The code in this repository is released under the [MIT License](LICENSE). The C++ components compiled from [Metasolver](https://github.com/OscarRojasG/Metasolver) are distributed under the GNU General Public License v3.0, and the training datasets on Zenodo under their own license.

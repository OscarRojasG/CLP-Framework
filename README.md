# CLP-Framework

A hybrid Transformer-based solver for the Container Loading Problem (CLP).

This solver utilizes the VCS heuristic by [Araya, Guerrero, and Nuñez](https://doi.org/10.1016/j.cor.2017.01.002) to identify the most promising blocks at each iteration. The Transformer model then performs a *re-ranking* of these candidate blocks, prioritizing those that lead to better overall solutions.

## Solver Variants

The framework implements both *greedy* and *beam search* procedures to evaluate the model. The following variants are available:

### 1. Greedy Search

* **GreedyModel:** Iteratively selects the most promising block based on the model's predictions.
* **VCS:** Uses the original VCS heuristic to select the next block (serves as the baseline).

### 2. Beam Search (BSG)

* **BSG-ModelVCS:** Uses the model to expand the best states at each tree level. To evaluate these states, it performs *greedy rollouts* using the VCS heuristic and selects the paths that yield the highest final volume.
* **BSG-ModelFull:** Uses the model for both state expansion and evaluation, performing *greedy rollouts* driven entirely by the model to select the highest final volume.
* **BSG-VCS:** A pure heuristic baseline that uses VCS for both state expansion and evaluation.

## Usage

For complete examples, refer to the `notebooks/validation.ipynb` notebook.

```python
model = load_model(CLPTransformer, model_name="M8")
input_adapter = InputAdapterV1(max_blocks=10000, max_pblocks=64, max_actions=64)

greedy_solver = GreedyModelSolver(model, w=8, input_adapter=input_adapter, min_fr=0.98)
greedy_eval, _ = greedy_solver.solve(instance_file, instance_number)
```

## Datasets

The datasets used for training each model can be found [here](https://drive.google.com/drive/folders/1TkktWgnxGwh_kNPzhdz6B3TCmYlHpLJu?usp=drive_link).

## Experiments

* [Greedy search results on BR1, BR8, and BR15 instances](https://docs.google.com/spreadsheets/d/e/2PACX-1vS2tIcPDaNfOZkr2Z-NQNAZY_IERS0cYhY91-JseMhhchmKU3zdXoVX0Px9XRlCkoowxIa8IbecGb9u/pubhtml)
* [Beam Search results on BR1, BR8, and BR15 instances](https://docs.google.com/spreadsheets/d/e/2PACX-1vRQau--eRIxDtcrFbOdBhkp_BvH8vug1BtO1ROXwepuhqTaMiQ2QwMD-_qnTNYCowjhugLN-E_bq010/pubhtml)
* [Results per model on the BR1-BR15 benchmarks](https://docs.google.com/spreadsheets/d/e/2PACX-1vQ_fTtYPZcPZXQLE-RV-k4WB4Xl_js0JUfyjnwtFilvkc17eRxEAjEBTxAdkrAT_X6etkwcHYLgva6S/pubhtml)
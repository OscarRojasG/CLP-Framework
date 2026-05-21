from torch.utils.data import Dataset, random_split, DataLoader, Subset
import torch
import os
import numpy as np
import copy
import json
import matplotlib.pyplot as plt
from settings import MODELS_FOLDER, HYPERPARAMETERS_FOLDER
from training.metrics import *
from utils import distribuir_suma_exacta


class Metrics:
    def __init__(self):
        self.loss_history = []
        self.acc_history = []

    def add_epoch(self, loss, acc):
        self.loss_history.append(loss)
        self.acc_history.append(acc)


class TrainingStats():
    def __init__(self):
        self.train_metrics : list[Metrics] = [] 
        self.val_metrics : list[list[Metrics]] = []
        self.phases = 0

    def add_phase_stats(self, train_metrics, val_metrics):
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        self.phases += 1

    def plot(self):
        # --- Concatenar accuracy de todas las fases ---
        train_acc = []
        val_accs = []
        phase_ends = []   # Guardar el punto donde termina cada fase
        total_epochs = 0
        
        for phase_idx in range(self.phases):
            phase_train = self.train_metrics[phase_idx].acc_history
            n_epochs = len(phase_train)

            # 1. Entrenamiento
            train_acc.extend(phase_train)
            
            # 2. Validaciones
            if phase_idx == 0:
                val_accs = [[] for _ in self.val_metrics[phase_idx]]
            
            for j, val in enumerate(self.val_metrics[phase_idx]):
                val_accs[j].extend(val.acc_history)

            # 3. Registrar el punto final de la fase
            total_epochs += n_epochs
            phase_ends.append(total_epochs)

        # --- Calcular accuracy promedio ---
        val_avg = np.mean(np.array(val_accs), axis=0) if val_accs else None

        # --- Graficar ---
        plt.figure(figsize=(8, 5))
        epochs = np.arange(1, len(train_acc) + 1)

        plt.plot(epochs, train_acc, label="Train Accuracy", linewidth=2)

        for i, val_acc in enumerate(val_accs):
            plt.plot(epochs, val_acc, linestyle="--", label=f"Val {self.val_metrics[0][i].subset_name}")

        if val_avg is not None:
            plt.plot(epochs, val_avg, label="Validation Avg", linewidth=2)

        # --- Líneas verticales para fases ---
        for x in phase_ends[:-1]:  # No marcar la última
            plt.axvline(x + 0.5, color="gray", linestyle="--", alpha=0.4)
            plt.text(x + 0.5, 2, f"Fase {phase_ends.index(x)+1}", rotation=90, 
                     va="bottom", ha="center", fontsize=8, alpha=0.6)

        plt.xlabel("Épocas")
        plt.ylabel("Accuracy (%)")
        plt.title("Historial de Accuracy")
        plt.ylim(0, 100)
        plt.legend(loc="center right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


class ValSubset(Subset):
    def __init__(self, subset, name):
        super().__init__(subset.dataset, subset.indices)
        self.name = name


# Modificado para entrelazar homogéneamente los datasets activos
class TrainSubset(Dataset):
    def __init__(self, subsets, names):
        self.subsets = subsets
        self.names = names
        self.num_datasets = len(subsets)
        # Como configuras tamaños fijos idénticos por dataset, tomamos el largo del primero
        self.single_len = len(subsets[0]) 
        self.total_len = self.single_len * self.num_datasets

    def __getitem__(self, index):
        # Determina a qué dataset corresponde el índice actual de forma cíclica
        dataset_idx = index % self.num_datasets
        # Determina la muestra interna correspondiente dentro de ese dataset
        sample_idx = index // self.num_datasets
        return self.subsets[dataset_idx][sample_idx]

    def __len__(self):
        return self.total_len


class DataManager:
    def __init__(self, datasets, train_size, test_size, seed):
        self.generator = torch.Generator().manual_seed(seed)
        self.processed_datasets = []
        
        # Particionamos cada dataset en sus tamaños definitivos desde el inicio
        for dataset in datasets:
            remainder = len(dataset) - train_size - test_size
            train_part, val_part, _ = random_split(
                dataset, 
                [train_size, test_size, remainder],
                generator=self.generator
            )
            
            self.processed_datasets.append({
                'train': train_part,
                'val': val_part,
                'name': dataset.name
            })

    def get_train_subset(self, phase):
        # Fase X acumula el entrenamiento de los primeros X datasets
        train_subsets = [self.processed_datasets[i]['train'] for i in range(phase)]
        dataset_names = [self.processed_datasets[i]['name'] for i in range(phase)]
        
        return TrainSubset(train_subsets, dataset_names)

    def get_val_subsets(self, phase):
        # Fase X devuelve una lista con las validaciones independientes de los primeros X datasets
        active_test_subsets = []
        for i in range(phase):
            entry = self.processed_datasets[i]
            active_test_subsets.append(ValSubset(subset=entry['val'], name=entry['name']))
            
        return active_test_subsets
    

class ModelScorer:
    def __init__(self, model):
        self.model = model
        self.best_models = {}

    def update_best_models(self, epoch, val_metrics: EpochMetrics):
        aux_dataset = list(val_metrics.subset_metrics.keys())[0]
        for metric in val_metrics.subset_metrics[aux_dataset].keys():
            sign = 1 if metric.maximize else -1
            score = sign * sum([val_metrics.subset_metrics[dataset][metric][-1] for dataset in val_metrics.subset_metrics.keys()]) / len(val_metrics.subset_metrics.keys())

            if metric in self.best_models and score < self.best_models[metric]["score"]: continue

            if metric not in self.best_models:
                self.best_models[metric] = {}
                
            self.best_models[metric]["score"] = score
            self.best_models[metric]["weights"] = copy.deepcopy(self.model.state_dict())
            self.best_models[metric]["epoch"] = epoch

    def print_best_scores(self):
        print("Mejores modelos por métrica:")
        for metric in self.best_models:
            sign = 1 if metric.maximize else -1
            print(f"    {metric.name}: {metric.format(sign * self.best_models[metric]['score'])} (Epoch {self.best_models[metric]['epoch']})")
        
    def get_best_weights(self):
        return {metric.name: self.best_models[metric]["weights"] for metric in self.best_models}
    
    def get_best_weights_by_metric(self, metric):
        return self.best_models[metric]["weights"]
    
    def get_last_update_epoch(self, metric):
        return self.best_models[metric]["epoch"]
    
    
def train_epoch(model, train_loader, optimizer, loss_function, metrics, device):
    model.train()

    for inputs, y_batch in train_loader:
        # Transferencia asíncrona
        inputs = [i.to(device, non_blocking=True) for i in inputs]
        targets = [t.to(device, non_blocking=True) for t in y_batch]

        logits = model(*inputs)

        total_loss = 0
        for target in targets:
            total_loss += loss_function.step(logits, target)

        for metric in metrics:
            for target in targets:
                metric.step(logits, target)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

    loss = loss_function.compute()
    values = [metric.compute() for metric in metrics]

    return loss, values


def val_epoch(model, val_loader, loss_function, metrics, device):
    model.eval()

    with torch.no_grad():
        for inputs_batch, y_batch in val_loader:
            # Desempaquetado dinámico para mayor flexibilidad
            inputs = [i.to(device, non_blocking=True) for i in inputs_batch]
            targets = [t.to(device, non_blocking=True) for t in y_batch]

            logits = model(*inputs)

            for target in targets:
                loss_function.step(logits, target)

            for metric in metrics:
                for target in targets:
                    metric.step(logits, target)

    loss = loss_function.compute()
    values = [metric.compute() for metric in metrics]

    return loss, values


def _train(model, epochs, train_set, test_sets, batch_size, learning_rate, weight_decay, loss_function, print_epoch_results, model_scorer: ModelScorer, patience, metrics, device):
    num_workers = os.cpu_count()
    
    # IMPORTANTE: shuffle=False para mantener nuestro orden entrelazado perfecto por batch.
    # No hay pérdida de aleatoriedad porque los subsets ya fueron mezclados por el random_split original.
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        persistent_workers=True
    )

    test_loaders = []
    for test_set in test_sets: 
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loaders.append(test_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_metrics = Metrics()

    train_metrics = EpochMetrics()
    val_metrics = EpochMetrics()
    
    for epoch in range(1, epochs+1):
        loss, values = train_epoch(model, train_loader, optimizer, loss_function, metrics, device)
        train_metrics.add_value("train", loss_function, loss)
        for i, value in enumerate(values):
            train_metrics.add_value("train", metrics[i], value)

        for test_loader in test_loaders:
            loss, values = val_epoch(model, test_loader, loss_function, metrics, device)
            for i, value in enumerate(values):
                val_metrics.add_value(test_loader.dataset.name, metrics[i], value)
            val_metrics.add_value(test_loader.dataset.name, loss_function, loss)

        print_epoch_results(epoch, train_metrics, val_metrics)
        model_scorer.update_best_models(epoch, val_metrics)

        if epoch - model_scorer.get_last_update_epoch(loss_function) > patience:
            break

    # Al terminar todas las fases, restauramos los mejores modelos
    weights = model_scorer.get_best_weights()
    model_scorer.print_best_scores()

    return weights, train_metrics, val_metrics

def train(model, epochs, datasets, train_size, test_size, batch_size, learning_rate, weight_decay, loss_function, patience, metrics, seed=42):
    data_manager = DataManager(datasets, train_size, test_size, seed)
    phases = len(datasets)

    ### CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"ℹ️ Usando dispositivo: {device}")

    torch.manual_seed(seed)
    torch.set_num_threads(os.cpu_count())
    model = model.to(device)

    for phase in range(1, phases+1):
        if epochs[phase-1] == 0: continue
        test_sets = data_manager.get_val_subsets(phase)
        train_set = data_manager.get_train_subset(phase)
        
        model_scorer = ModelScorer(model)

        def print_epoch_results(epoch, train_metrics, val_metrics):
            print(f'{'\n' if epoch == 1 else ''}Epoch {epoch}/{epochs[phase-1]}')
            val_epoch_loss = sum([val_metrics.subset_metrics[dataset][loss_function][-1] for dataset in val_metrics.subset_metrics.keys()]) / phase
            train_epoch_loss = train_metrics.subset_metrics["train"][loss_function][-1]
            print(f"    Average - Train Loss: {loss_function.format(train_epoch_loss)} | Val Loss: {loss_function.format(val_epoch_loss)}", end='')

            for metric in train_metrics.subset_metrics["train"]:
                if metric == loss_function: continue
                val_epoch_metric = 0
                for i, dataset in enumerate(val_metrics.subset_metrics.keys()):
                    val_epoch_metric += val_metrics.subset_metrics[dataset][metric][-1] / phase
                print(f' | {metric.name}: {metric.format(val_epoch_metric)}', end='')
            print()

            for i, dataset in enumerate(val_metrics.subset_metrics.keys()):
                print(f"    Dataset {dataset} - ", end='')
                for j, metric in enumerate(val_metrics.subset_metrics[dataset]):
                    if metric != loss_function:
                        val_epoch_metric = val_metrics.subset_metrics[dataset][metric][-1]
                        print(f'{' | ' if j > 0 else ''}{metric.name}: {metric.format(val_epoch_metric)}', end='')
                print()

        print(f"{'\n' if phase > 1 else ''}ℹ️ Iniciando fase: {phase}/{phases}")
        best_weights, train_metrics, val_metrics = _train(model, epochs[phase-1], train_set, test_sets, batch_size, learning_rate[phase-1], weight_decay, loss_function, print_epoch_results, model_scorer, patience, metrics, device)
        weights = model_scorer.get_best_weights_by_metric(loss_function)
        model.load_state_dict(weights)

        #stats.add_phase_stats(train_metrics, val_metrics)

    return model

def save_model(model, model_name):
    os.makedirs(HYPERPARAMETERS_FOLDER, exist_ok=True)
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'w') as f:
        json.dump(model.hyperparams, f, indent=4)

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    weights = model.state_dict()
    torch.save(weights, str(MODELS_FOLDER / model_name) + ".pth")
    print(f"✅ Modelo guardado en {MODELS_FOLDER / model_name}.pth")

def load_hyperparams(model_name):
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'r') as f:
        return json.load(f)

def load_model(model_class: object, model_name):
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'r') as f:
        hyperparams = json.load(f)

    model = model_class(**hyperparams)
    model.load_state_dict(torch.load(str(MODELS_FOLDER / model_name) + ".pth", weights_only=True, map_location=torch.device('cpu')), strict=True)
    model.eval()
    return model
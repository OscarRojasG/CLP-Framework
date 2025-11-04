from torch import nn
from torch.utils.data import TensorDataset, random_split, ConcatDataset, DataLoader, Subset
import torch
import os
import numpy as np
import copy
from src.data_generator import load_data_from_file
from src.data_preprocessing import normalize_input, feature_expansion
import matplotlib.pyplot as plt
from .models.base.encoder_decoder import EncoderDecoderModel
from .models.base.encoder_decoder_pe import EncoderDecoderPEModel

model_folder_path = "models/"


class Metrics:
    def __init__(self):
        self.loss_history = []
        self.acc_history = []
        self.reset_epoch()

    def reset_epoch(self):
        self.loss_sum = 0
        self.correct = 0
        self.total = 0

    def update_batch(self, outputs, targets, loss, batch_size):
        self.loss_sum += loss * batch_size
        self.total += batch_size
        _, predicted = torch.max(outputs.data, 1)
        self.correct += (predicted == targets.argmax(dim=1)).sum().item()

    def end_epoch(self):
        epoch_loss = self.loss_sum / self.total if self.total > 0 else 0
        epoch_acc = 100 * self.correct / self.total if self.total > 0 else 0
        self.loss_history.append(epoch_loss)
        self.acc_history.append(epoch_acc)
        self.reset_epoch()

class TrainingStats():
    def __init__(self):
        self.train_metrics : list[Metrics] = [] 
        self.val_metrics : list[list[ValMetrics]] = []
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

class ValMetrics(Metrics):
    def __init__(self, subset_name):
        super().__init__()
        self.subset_name = subset_name

class ValSubset(Subset):
    def __init__(self, subset, name):
        super().__init__(subset.dataset, subset.indices)
        self.name = name

class TrainSubset(ConcatDataset):
    def __init__(self, subsets, origin_dataset_names):
        super().__init__(subsets)
        self.origin_dataset_names = origin_dataset_names

class NamedDataset(TensorDataset):
    def __init__(self, name, *tensors):
        super().__init__(*tensors)
        self.name = name


def save_model(model: nn.Module, filename):
    os.makedirs(model_folder_path, exist_ok=True)
    torch.save(model.state_dict(), model_folder_path + filename)

def load_model(empty_model: nn.Module, filename):
    empty_model.load_state_dict(torch.load(model_folder_path + filename, weights_only=True))
    empty_model.eval()
    return empty_model

def load_dataset(dataset_file):
    X_src, X_tgt, Y, placed, coords = load_data_from_file(dataset_file)
    X_src = torch.tensor(X_src, dtype=torch.float32)
    X_tgt = torch.tensor(X_tgt, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    placed = torch.tensor(placed, dtype=torch.float32)
    coords = torch.tensor(coords, dtype=torch.float32)
    return NamedDataset(dataset_file, X_src, X_tgt, Y, placed, coords)

def create_train_subsets(train_datasets, train_size, train_weights, seed=42):
    torch.manual_seed(seed)
    train_subsets = []
    dataset_names = []

    for i, train_dataset in enumerate(train_datasets):
        subset_size = int(train_size * train_weights[i] / sum(train_weights))
        if subset_size == 0: continue
        train_subset, _ = random_split(train_dataset, [subset_size, len(train_dataset) - subset_size])
        train_subsets.append(train_subset)
        dataset_names.append(train_dataset.name)

    return TrainSubset(subsets=train_subsets, origin_dataset_names=dataset_names)

def create_subsets(datasets, train_size, train_weights, test_size, test_weights, seed):
    torch.manual_seed(seed)

    test_subsets = []
    train_datasets = []

    for i, dataset in enumerate(datasets):
        subset_size = int(test_size * test_weights[i] / sum(test_weights))
        test_subset, complement_set = random_split(dataset, [subset_size, len(dataset) - subset_size])
        test_subsets.append(ValSubset(subset=test_subset, name=dataset.name))
        complement_set = ValSubset(subset=complement_set, name=dataset.name)
        train_datasets.append(complement_set)
    
    train_subset = create_train_subsets(train_datasets, train_size, train_weights)
    
    return train_subset, test_subsets

def get_predictions(model, X_src_batch, X_tgt_batch, placed_batch, coords_batch, apply_softmax=False):
    if isinstance(model, EncoderDecoderPEModel):
        return model.predict(X_src_batch, X_tgt_batch, placed_batch, coords_batch, apply_softmax)
    if isinstance(model, EncoderDecoderModel):
        return model.predict(X_src_batch, X_tgt_batch, apply_softmax)
    else:
        return model.predict(X_tgt_batch, apply_softmax)

def _train(model, epochs, train_set, test_sets, batch_size, learning_rate, patience, seed=42) -> tuple[nn.Module, Metrics, list[Metrics]]:
    # --- CONFIGURAR DISPOSITIVO ---
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print(f"Usando dispositivo: {device}")

    torch.manual_seed(seed)
    torch.set_num_threads(os.cpu_count())

    # Mover el modelo al dispositivo
    model = model.to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    test_loaders = []
    for test_set in test_sets: 
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
        test_loaders.append((test_loader, ValMetrics(test_set.name)))

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_metrics = Metrics()

    # Early stopping
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(epochs):
        # --- ENTRENAMIENTO ---
        model.train()
        for X_src_batch, X_tgt_batch, y_batch, placed_batch, coords_batch in train_loader:
            # Mover los datos al dispositivo
            X_src_batch = X_src_batch.to(device)
            X_tgt_batch = X_tgt_batch.to(device)
            placed_batch = placed_batch.to(device)
            coords_batch = coords_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = get_predictions(model, X_src_batch, X_tgt_batch, placed_batch, coords_batch)
            loss = loss_function(outputs, y_batch.argmax(dim=-1))
            loss.backward()
            optimizer.step()

            train_metrics.update_batch(outputs, y_batch, loss.item(), X_tgt_batch.size(0))

        train_metrics.end_epoch()

        # --- VALIDACIÓN ---
        model.eval()
        with torch.no_grad():
            for test_loader, metrics in test_loaders:
                for X_src_batch, X_tgt_batch, y_batch, placed_batch, coords_batch in test_loader:
                    X_src_batch = X_src_batch.to(device)
                    X_tgt_batch = X_tgt_batch.to(device)
                    placed_batch = placed_batch.to(device)
                    coords_batch = coords_batch.to(device)
                    y_batch = y_batch.to(device)

                    outputs = get_predictions(model, X_src_batch, X_tgt_batch, placed_batch, coords_batch)
                    loss = loss_function(outputs, y_batch.argmax(dim=-1))
                    metrics.update_batch(outputs, y_batch, loss.item(), X_tgt_batch.size(0))
                metrics.end_epoch()
            
        # --- PRINT EPOCH RESULTS ---
        samples_per_set = [len(test_set) for test_set in test_sets]
        overall_weights = [samples / sum(samples_per_set) for samples in samples_per_set]

        samples_per_set = [len(test_set) if test_set.name in train_set.origin_dataset_names else 0 for test_set in test_sets]
        epoch_weights = [samples / sum(samples_per_set) for samples in samples_per_set]

        val_loss_wgt = np.sum(test_loaders[i][1].loss_history[-1] * overall_weights[i] for i in range(len(test_loaders)))
        val_acc_wgt = np.sum(test_loaders[i][1].acc_history[-1] * overall_weights[i] for i in range(len(test_loaders)))

        val_loss_epoch = np.sum(test_loaders[i][1].loss_history[-1] * epoch_weights[i] for i in range(len(test_loaders)) if test_loaders[i][1].subset_name in train_set.origin_dataset_names)
        val_acc_epoch = np.sum(test_loaders[i][1].acc_history[-1] * epoch_weights[i] for i in range(len(test_loaders)) if test_loaders[i][1].subset_name in train_set.origin_dataset_names)

        print(f'Epoch {epoch + 1}/{epochs} - '
            f'Train Loss: {train_metrics.loss_history[-1]:.4f}, '
            f'Train Accuracy: {train_metrics.acc_history[-1]:.2f}% - '
            f'Val Loss: {val_loss_epoch:.4f}, Val Accuracy: {val_acc_epoch:.2f}%')

        for _, m in test_loaders:
            print(f'    Test Set: {m.subset_name} - '
                f'Val Loss: {m.loss_history[-1]:.4f}, Val Accuracy: {m.acc_history[-1]:.2f}%')

        print(f'    Weighted - '
            f'Val Loss: {val_loss_wgt:.4f}, Val Accuracy: {val_acc_wgt:.2f}%')  

        # --- EARLY STOPPING SOLO EN LOSS ---
        if val_loss_epoch < best_loss:
            best_loss = val_loss_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping en la epoch {epoch+1}")
                model.load_state_dict(best_model_wts)
                return model, train_metrics, [metrics for _, metrics in test_loaders]

    # Al terminar todas las fases, restauramos el mejor modelo
    model.load_state_dict(best_model_wts)
    return model, train_metrics, [metrics for _, metrics in test_loaders]

def train(model, epochs, datasets, train_size, train_weights, test_size, test_weights, batch_size, learning_rate, patience, seed=42):
    train_set, test_sets = create_subsets(datasets, train_size, train_weights, test_size, test_weights, seed)
    stats = TrainingStats()

    model, train_metrics, val_metrics = _train(model, epochs, train_set, test_sets, batch_size, learning_rate, patience, seed)
    stats.add_phase_stats(train_metrics, val_metrics)

    return stats
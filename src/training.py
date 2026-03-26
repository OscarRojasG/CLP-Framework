from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader, Subset
import torch
import os
import numpy as np
import copy
import json
import matplotlib.pyplot as plt
from settings import MODELS_FOLDER, HYPERPARAMS_FOLDER
from torch.amp import GradScaler, autocast


class Metrics:
    def __init__(self):
        self.loss_history = []
        self.acc_history = []

    def add_epoch(self, loss, acc):
        self.loss_history.append(loss)
        self.acc_history.append(acc)


class ValMetrics(Metrics):
    def __init__(self, subset_name):
        super().__init__()
        self.subset_name = subset_name


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


class ValSubset(Subset):
    def __init__(self, subset, name):
        super().__init__(subset.dataset, subset.indices)
        self.name = name


class TrainSubset(Dataset):
    def __init__(self, subsets, names, weights):
        self.dataset = torch.utils.data.ConcatDataset(subsets)
        self.names = names
        
        sample_weights = []
        for i, subset in enumerate(subsets):
            sample_weights.extend([weights[i]] * len(subset))
            
        self.sample_weights = torch.DoubleTensor(sample_weights)

    def __getitem__(self, index):
        data = self.dataset[index]
        return data

    def __len__(self):
        return len(self.dataset)


class DataManager:
    def __init__(self, datasets, train_size, train_weights, test_size, test_weights, seed):
        self.datasets = datasets
        self.train_size = train_size
        self.train_weights = train_weights
        self.test_size = test_size
        self.test_weights = test_weights
        self.seed = seed
        
        # Guardamos los datasets originales para sacar subsets frescos en cada fase
        self.all_datasets = datasets

    def get_val_subsets(self, phase):
        torch.manual_seed(self.seed + phase)
        active_test_subsets = []
        
        active_weights = self.test_weights[:phase]
        total_test_w = sum(active_weights)

        for i in range(phase):
            dataset = self.all_datasets[i]
            # Calculamos cuánto le toca a este dataset de la cuota total
            subset_size = int(self.test_size * active_weights[i] / total_test_w)
            
            if subset_size == 0: continue
            
            # Sacamos el subset de validación
            # Nota: No usamos complement_set aquí porque lo gestionamos en get_train_subset
            val_subset, _ = random_split(
                dataset, [subset_size, len(dataset) - subset_size]
            )
            active_test_subsets.append(ValSubset(subset=val_subset, name=dataset.name))
            
        return active_test_subsets

    def get_train_subset(self, phase):
        torch.manual_seed(self.seed + phase)
        train_subsets = []
        dataset_names = []
        
        active_train_weights = self.train_weights[:phase]
        active_test_weights = self.test_weights[:phase]
        
        total_tw = sum(active_train_weights)
        total_test_w = sum(active_test_weights)

        for i in range(phase):
            dataset = self.all_datasets[i]
            
            # Identificamos qué datos son de TEST para excluirlos
            test_size_i = int(self.test_size * active_test_weights[i] / total_test_w)
            val_subset, complement_set = random_split(
                dataset, [test_size_i, len(dataset) - test_size_i]
            )
            
            train_size_i = int(self.train_size * active_train_weights[i] / total_tw)
            if train_size_i == 0: continue
            
            # Si el complement_set es más pequeño que la cuota pedida, usamos todo el complement
            actual_train_size = min(train_size_i, len(complement_set))
            
            train_subset, _ = random_split(
                complement_set, [actual_train_size, len(complement_set) - actual_train_size]
            )
            
            train_subsets.append(train_subset)
            dataset_names.append(dataset.name)

        return TrainSubset(train_subsets, dataset_names, active_train_weights)
    

class ModelScorer:
    def __init__(self, model, score_function, print_best_score):
        self.best_score = float("-inf")
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.model = model
        self.score_function = score_function
        self._print_best_score = print_best_score
    
    def update_best_model(self, val_metrics_list):
        score = self.score_function(val_metrics_list)
        if score > self.best_score:
            #print(f"Mejor modelo actualizado: {self.best_score:2f} -> {score:2f}")
            self.best_score = score
            self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def print_best_score(self):
        self._print_best_score(self.best_score)
        
    def get_best_model(self):
        self.model.load_state_dict(self.best_model_wts)
        return self.model
    
    
def train_epoch(model, train_loader, loss_function, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for *inputs, y_batch in train_loader:
        # Transferencia asíncrona
        inputs = [i.to(device, non_blocking=True) for i in inputs]
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) # Más eficiente que zero_grad()

        # Autocast para precisión mixta (FP16)
        with autocast(device.type):
            outputs = model(*inputs)
            labels = y_batch.argmax(dim=-1)
            loss = loss_function(outputs, labels) 

        # Escalamiento de gradientes para evitar subdesbordamiento (underflow)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Métricas (Intentamos minimizar el paso de datos GPU -> CPU)
        with torch.no_grad():
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples

    return loss, accuracy

def val_epoch(model: nn.Module, val_loader: DataLoader, loss_function, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Usamos autocast también en validación para que las activaciones 
    # tengan el mismo formato que en el entrenamiento
    with torch.no_grad(), autocast(device.type):
        for batch in val_loader:
            # Desempaquetado dinámico para mayor flexibilidad
            *inputs, y_batch = [i.to(device, non_blocking=True) for i in batch]

            outputs = model(*inputs)
            labels = y_batch.argmax(dim=-1)
            loss = loss_function(outputs, labels)

            # Métricas
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples

    return loss, accuracy

def _train(model, epochs, train_set, test_sets, batch_size, learning_rate, weight_decay, print_epoch_results, model_scorer, patience, device): 
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_set.sample_weights, 
        num_samples=len(train_set), 
        replacement=True
    )

    num_workers = os.cpu_count()
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        persistent_workers=True
    )
    
    scaler = GradScaler(device.type)

    test_loaders = []
    val_metrics_list = []
    for test_set in test_sets: 
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loaders.append(test_loader)
        val_metrics_list.append(ValMetrics(test_set.name))

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_metrics = Metrics()

    epochs_without_improvement = 0
    best_score = float("-inf")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, loss_function, optimizer, device, scaler)
        train_metrics.add_epoch(train_loss, train_acc)

        for test_loader, val_metrics in zip(test_loaders, val_metrics_list):
            val_loss, val_acc = val_epoch(model, test_loader, loss_function, device)
            val_metrics.add_epoch(val_loss, val_acc)

        print_epoch_results(epoch, train_metrics, val_metrics_list)
        model_scorer.update_best_model(val_metrics_list)
        
        if best_score == model_scorer.best_score:
            epochs_without_improvement += 1
        else:
            best_score = model_scorer.best_score
            epochs_without_improvement = 0

        if epochs_without_improvement > patience:
            break

    # Al terminar todas las fases, restauramos el mejor modelo
    model = model_scorer.get_best_model()
    model_scorer.print_best_score()

    return model, train_metrics, val_metrics_list

def train(model, epochs, datasets, train_size, train_weights, test_size, test_weights, batch_size, learning_rate, weight_decay, patience, seed=42):
    data_manager = DataManager(datasets, train_size, train_weights, test_size, test_weights, seed)
    stats = TrainingStats()
    phases = len(datasets)
    
    def print_best_score(best_score):
        print(f"✅ Mejor loss obtenido: {-best_score:.4f}\n")

    ### CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print(f"ℹ️ Usando dispositivo: {device}")

    torch.manual_seed(seed)
    torch.set_num_threads(os.cpu_count())
    model = model.to(device)


    for phase in range(1, phases+1):
        if epochs[phase-1] == 0: continue
        test_sets = data_manager.get_val_subsets(phase)
        train_set = data_manager.get_train_subset(phase)

        samples_per_set = [len(test_set) for test_set in test_sets]
        epoch_weights = [samples / sum(samples_per_set) for samples in samples_per_set]

        def score_function(val_metrics_list):
            return -sum([val_metrics.loss_history[-1] * epoch_weights[i] / sum(epoch_weights) for i, val_metrics in enumerate(val_metrics_list)])
        
        model_scorer = ModelScorer(model, score_function, print_best_score)

        def print_epoch_results(epoch, train_metrics, val_metrics_list):
            val_epoch_loss = sum([val_metrics.loss_history[-1] * epoch_weights[i] / sum(epoch_weights) for i, val_metrics in enumerate(val_metrics_list)])
            val_epoch_acc = sum([val_metrics.acc_history[-1] * epoch_weights[i] / sum(epoch_weights) for i, val_metrics in enumerate(val_metrics_list)])

            val_wgt_loss = sum([val_metrics.loss_history[-1] * test_weights[i] / sum(test_weights) for i, val_metrics in enumerate(val_metrics_list)])
            val_wgt_acc = sum([val_metrics.acc_history[-1] * test_weights[i] / sum(test_weights) for i, val_metrics in enumerate(val_metrics_list)])

            print(f'Epoch {epoch + 1}/{epochs[phase-1]} - '
                f'Train Loss: {train_metrics.loss_history[-1]:.4f}, '
                f'Train Accuracy: {train_metrics.acc_history[-1]:.2f}% - '
                f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}%')

            for i, val_metrics in enumerate(val_metrics_list):
                print(f'    Test Set: {val_metrics.subset_name} - '
                    f'Val Loss: {val_metrics.loss_history[-1]:.4f}, Val Accuracy: {val_metrics.acc_history[-1]:.2f}%')

            print(f'    Weighted - '
                f'Val Loss: {val_wgt_loss:.4f}, Val Accuracy: {val_wgt_acc:.2f}%')

        print(f"ℹ️ Iniciando fase: {phase}/{phases}")
        model, train_metrics, val_metrics = _train(model, epochs[phase-1], train_set, test_sets, batch_size, learning_rate, weight_decay, print_epoch_results, model_scorer, patience, device)
        stats.add_phase_stats(train_metrics, val_metrics)

    return stats

def save_model(model, model_name):
    os.makedirs(HYPERPARAMS_FOLDER, exist_ok=True)
    with open(str(HYPERPARAMS_FOLDER / model_name) + ".json", 'w') as f:
        json.dump(model.hyperparams, f, indent=4)

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    torch.save(model.state_dict(), str(MODELS_FOLDER / model_name) + ".pth")
    print(f"✅ Modelo guardado en {MODELS_FOLDER / model_name}.pth")

def load_hyperparams(model_name):
    with open(str(HYPERPARAMS_FOLDER / model_name) + ".json", 'r') as f:
        return json.load(f)

def load_model(model_class: object, model_name):
    with open(str(HYPERPARAMS_FOLDER / model_name) + ".json", 'r') as f:
        hyperparams = json.load(f)

    model = model_class(**hyperparams)
    model.load_state_dict(torch.load(str(MODELS_FOLDER / model_name) + ".pth", weights_only=True, map_location=torch.device('cpu')), strict=True)
    model.eval()
    return model
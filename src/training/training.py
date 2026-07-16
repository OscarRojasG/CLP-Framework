import copy
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from training.logging import save_experiment_config, save_phase_history
from training.metrics import *
from settings import MODELS_FOLDER, HYPERPARAMETERS_FOLDER, EXPERIMENTS_FOLDER
import os
import json
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataclasses import dataclass, asdict

@dataclass
class LRConfig:
    start: float            # Tasa de aprendizaje inicial
    factor: float = 0.5     # Factor de reducción
    patience: int = 999999  # Épocas sin mejora antes de reducir el LR
    min: float = 0.0        # Tasa de aprendizaje mínima permitida


def compute_weighted_average(metrics_dict, samples_dict):
    """
    metrics_dict: {ds_name: {metric_name: value}}
    samples_dict: {ds_name: count}
    """
    total_samples = sum(samples_dict.values())
    if total_samples == 0: return {}

    # Obtenemos los nombres de las métricas de la primera entrada
    first_ds = next(iter(metrics_dict))
    metric_names = metrics_dict[first_ds].keys()

    global_metrics = {}
    for m_name in metric_names:
        weighted_sum = sum(
            metrics_dict[ds][m_name] * samples_dict[ds]
            for ds in metrics_dict
        )
        global_metrics[m_name] = weighted_sum / total_samples

    return global_metrics


# --- 1. Motor de Entrenamiento (TrainerEngine) ---

class TrainerEngine:
    def __init__(self, model, device, loss_fn, metrics: list, weights: list, dataset_names: list):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.base_metrics = metrics
        self.weights = torch.tensor(weights, device=device)
        self.dataset_names = dataset_names

        # Estado persistente
        self.metric_trackers = {}
        self.loss_trackers = {}

    def _get_dataset_metrics(self, ds_name):
        if ds_name not in self.metric_trackers:
            self.metric_trackers[ds_name] = [copy.deepcopy(m) for m in self.base_metrics]
        return self.metric_trackers[ds_name]

    def _get_dataset_loss(self, ds_name):
        if ds_name not in self.loss_trackers:
            self.loss_trackers[ds_name] = copy.deepcopy(self.loss_fn)
        return self.loss_trackers[ds_name]

    def run_epoch(self, loader, optimizer=None):
        is_train = optimizer is not None
        self.model.train() if is_train else self.model.eval()
        context = torch.enable_grad() if is_train else torch.no_grad()

        # Resetear estado al inicio de cada epoch
        epoch_losses = defaultdict(float)
        epoch_samples = defaultdict(int)  # ← local, no persistente

        # Resetear metric trackers
        for trackers in self.metric_trackers.values():
            for m in trackers:
                m.reset()

        with context:
            for inputs_batch, targets_batch, ids_batch in loader:
                unique_ids = torch.unique(ids_batch)
                batch_loss_total = 0

                for ds_id in unique_ids:
                    ds_idx = ds_id.item()
                    ds_name = self.dataset_names[ds_idx]

                    mask = (ids_batch == ds_id)
                    ds_inputs = [i[mask].to(self.device, non_blocking=True) for i in inputs_batch]
                    ds_targets = [t[mask].to(self.device, non_blocking=True) for t in targets_batch]

                    logits = self.model(*ds_inputs)
                    ds_loss_tracker = self._get_dataset_loss(ds_name)
                    ds_loss = ds_loss_tracker.step(logits, ds_targets[0])  # acumula correctamente

                    epoch_losses[ds_name] += ds_loss.item()
                    epoch_samples[ds_name] += mask.sum().item()  # ← local

                    weight = self.weights[ds_idx] if self.weights is not None else 1.0
                    batch_loss_total += ds_loss * weight

                    ds_metrics = self._get_dataset_metrics(ds_name)
                    for metric in ds_metrics:
                        for t in ds_targets:
                            metric.step(logits, t)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss_total.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

        # Cálculo final con samples locales
        final_losses = {
            name: self.loss_trackers[name].compute()
            for name in epoch_samples
        }
        final_metrics = {
            name: {m.name: m.compute() for m in self.metric_trackers[name]}
            for name in epoch_losses
        }

        return final_losses, final_metrics, epoch_samples  # ← devuelve samples locales


# --- 2. Reporter (Manejo de Impresión) ---

class TrainingReporter:
    @staticmethod
    def print_epoch(epoch, total_epochs, train_metrics, val_metrics, loss_fn, metrics_list, weighted_metrics, val_losses, current_train_loss, current_val_loss):
        print(f"\nEpoch {epoch}/{total_epochs}")

        # Obtener nombres de datasets
        train_ds_names = list(train_metrics.subset_metrics.keys())
        val_ds_names = list(val_metrics.subset_metrics.keys())

        # Imprimimos directamente los losses ponderados calculados en run_phase
        print(f"    Global - Wgt Train Loss: {loss_fn.format(current_train_loss)} | Wgt Val Loss: {loss_fn.format(current_val_loss)}")

        # Mostrar detalle por dataset (Entrenamiento)
        for ds_name in train_ds_names:
            loss = train_metrics.get_last_metric_value(ds_name, loss_fn)
            print(f"    [Train] {ds_name} - Loss: {loss_fn.format(loss)}")

        # Mostrar detalle por dataset (Validación)
        for ds_name in val_ds_names:
            loss = val_losses[ds_name]  # ← directo del dict
            print(f"    [Val] {ds_name} - Loss: {loss_fn.format(loss)} | ", end='')
            for metric in metrics_list:
                try:
                    val = val_metrics.get_last_metric_value(ds_name, metric)
                    print(f"{metric.name}: {metric.format(val)} | ", end='')
                except KeyError:
                    print(f"{metric.name}: N/A | ", end='')
            print()

        print("    [Avg] ", end='')
        for m_name, value in weighted_metrics.items():
            metric_obj = next((m for m in metrics_list if m.name == m_name), None)
            print(f"{m_name}: {metric_obj.format(value)} | ", end='')
        print()

# --- 3. DataManager ---

class ValSubset(Subset):
    def __init__(self, subset, name, dataset_id):
        super().__init__(subset.dataset, subset.indices)
        self.name = name
        self.dataset_id = dataset_id

    def __getitem__(self, idx):
        inputs, targets = super().__getitem__(idx)
        return inputs, targets, self.dataset_id

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

class TrainSubset(Dataset):
    def __init__(self, subsets, names, dataset_ids):
        self.subsets = subsets
        self.names = names
        self.dataset_ids = dataset_ids
        self.num_datasets = len(subsets)
        self.single_len = len(subsets[0])
        self.total_len = self.single_len * self.num_datasets

    def __getitem__(self, index):
        dataset_idx = index % self.num_datasets
        sample_idx = index // self.num_datasets
        inputs, targets = self.subsets[dataset_idx][sample_idx]
        return inputs, targets, self.dataset_ids[dataset_idx]

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

        # Generamos los IDs (0, 1, ..., phase-1)
        dataset_ids = list(range(phase))

        return TrainSubset(train_subsets, dataset_names, dataset_ids)

    def get_val_subsets(self, phase):
        active_test_subsets = []
        for i in range(phase):
            entry = self.processed_datasets[i]
            active_test_subsets.append(ValSubset(subset=entry['val'], name=entry['name'], dataset_id=i))
        return active_test_subsets

# --- 4. Entrenamiento ---

def run_phase(model, train_loader, val_loaders, epochs, optimizer, scheduler, loss_fn, weights, dataset_names, metrics, patience, device):
    engine = TrainerEngine(model, device, loss_fn, metrics, weights, dataset_names)
    train_metrics = EpochMetrics()
    val_metrics = EpochMetrics()

    best_val_loss = float('inf')
    best_weights = None
    epochs_without_improvement = 0
    phase_history = {}

    for epoch in range(1, epochs + 1):
        # 1. Entrenamiento: Reseteamos antes de empezar
        t_losses, t_metrics, _ = engine.run_epoch(train_loader, optimizer)

        # Registrar entrenamiento
        for ds_name, loss in t_losses.items():
            train_metrics.add_value(ds_name, loss_fn, loss)
        for ds_name, values in t_metrics.items():
            for m_name, val in values.items(): # values es dict ahora
                train_metrics.add_value(ds_name, next(m for m in metrics if m.name == m_name), val)

        # 2. Validación: Reseteamos antes de iterar TODOS los loaders
        epoch_val_losses = {}
        epoch_val_metrics = {}
        epoch_val_samples = {}

        for loader in val_loaders:
            v_losses_dict, v_vals_dict, v_samples_dict = engine.run_epoch(loader)
            ds_name = loader.dataset.name

            epoch_val_losses[ds_name] = v_losses_dict[ds_name]
            epoch_val_metrics[ds_name] = v_vals_dict[ds_name]
            epoch_val_samples[ds_name] = v_samples_dict[ds_name]

            val_metrics.add_value(ds_name, loss_fn, v_losses_dict[ds_name])
            for m_name, val in v_vals_dict[ds_name].items():
                val_metrics.add_value(ds_name, next(m for m in metrics if m.name == m_name), val)

        # 3. Reporte y Scheduler
        # Calculamos AMBOS losses ponderados usando los pesos normalizados
        current_train_loss = sum(t_losses[name] * w for name, w in zip(dataset_names, weights))
        current_val_loss = sum(epoch_val_losses[name] * w for name, w in zip(dataset_names, weights))
        
        global_metrics = compute_weighted_average(epoch_val_metrics, epoch_val_samples)

        if scheduler:
            scheduler.step(current_val_loss) if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()

        # Pasamos ambos losses ponderados al final
        TrainingReporter.print_epoch(
            epoch, epochs, train_metrics, val_metrics,
            loss_fn, metrics, global_metrics, epoch_val_losses,
            current_train_loss, current_val_loss  # ← Ahora pasamos los dos
        )

        phase_history[epoch] = {"train": {"losses": t_losses, "metrics": t_metrics}, "val": {"losses": epoch_val_losses, "metrics": epoch_val_metrics}}

        # Checkpointing
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_weights:
        model.load_state_dict(best_weights)

    return best_weights, phase_history


def train(model, epochs, datasets, train_size, test_size, batch_size,
          lr_configs, weight_decay, loss_function, loss_weights, patience, metrics, seed=42):

    # --- PREPARACIÓN DE LA CONFIGURACIÓN ---
    config = {
        "epochs": epochs,
        "datasets": [d.name for d in datasets],
        "train_size": train_size,
        "test_size": test_size,
        "batch_size": batch_size,
        # Guardamos la configuración completa de LR para cada fase
        "lr_configs": [asdict(cfg) for cfg in lr_configs], 
        "weight_decay": weight_decay,
        "loss_function": loss_function.name,
        "loss_weights": loss_weights,
        "patience": patience,
        "metrics": [m.name for m in metrics],
        "seed": seed,
    }

    # Guardamos la configuración al iniciar
    os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)
    log_path = EXPERIMENTS_FOLDER / "experiment_logs.json"
    save_experiment_config(log_path, config)

    # 1. Preparación del ambiente
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"ℹ️ Usando dispositivo: {device}")

    torch.manual_seed(seed)
    model = model.to(device)
    data_manager = DataManager(datasets, train_size, test_size, seed)

    # 2. Bucle de Fases
    phases = len(datasets)
    for phase in range(1, phases + 1):
        if epochs[phase-1] == 0: continue

        # Normalización: solo datasets activos (hasta 'phase')
        active_weights = loss_weights[:phase]
        total_w = sum(active_weights)
        norm_weights = [w / total_w for w in active_weights]

        active_names = [d.name for d in datasets[:phase]]

        print(f"\nℹ️ Iniciando fase: {phase}/{phases}")

        train_loader = DataLoader(
            data_manager.get_train_subset(phase),
            batch_size=batch_size, shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=(device.type == "cuda")
        )

        val_subsets = data_manager.get_val_subsets(phase)
        val_loaders = [
            DataLoader(subset, batch_size=batch_size, shuffle=False)
            for subset in val_subsets
        ]

        # Extraemos la configuración de LR correspondiente a esta fase
        current_lr_config = lr_configs[phase-1]

        # Configurar optimizador específico de la fase
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=current_lr_config.start,
            weight_decay=weight_decay
        )

        # Configurar el scheduler con la clase
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=current_lr_config.factor,
            patience=current_lr_config.patience,
            min_lr=current_lr_config.min
        )

        # 3. Ejecutar fase
        best_weights, phase_history = run_phase(
            model=model,
            train_loader=train_loader,
            val_loaders=val_loaders,
            epochs=epochs[phase-1],
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_function,
            weights=norm_weights,
            dataset_names=active_names,
            metrics=metrics,
            patience=patience,
            device=device
        )

        # Guardar en JSON
        save_phase_history(log_path, phase, phase_history)

        # Cargar los mejores pesos al terminar la fase
        model.load_state_dict(best_weights)

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
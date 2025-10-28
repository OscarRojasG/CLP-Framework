from torch import nn
import torch
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, ConcatDataset, DataLoader, Subset
import copy
from src.data_generator import load_data_from_file

model_folder_path = "models/"


class Metrics():
    def __init__(self):
        self.loss_history = []
        self.acc_history = []
        self.loss = 0
        self.total = 0
        self.correct = 0

    def update_loss(self, loss, batch_size):
        self.loss += loss * batch_size
        self.total += batch_size
        self.loss_history.append(self.loss / self.total)

    def update_accuracy(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        self.correct += (predicted == targets.argmax(dim=1)).sum().item()
        accuracy = 100 * self.correct / self.total
        self.acc_history.append(accuracy)

class NamedSubset(Subset):
    def __init__(self, subset, name):
        super().__init__(subset.dataset, subset.indices)
        self.name = name

def save_model(model: nn.Module, filename):
    os.makedirs(model_folder_path, exist_ok=True)
    torch.save(model.state_dict(), model_folder_path + filename)

def load_model(empty_model: nn.Module, filename):
    empty_model.load_state_dict(torch.load(model_folder_path + filename, weights_only=True))
    empty_model.eval()
    return empty_model

def normalize_input(X):
    # Escalar con StandardScaler
    # X shape: [num_ejemplos, num_acciones, 4]
    X = np.array(X, dtype=np.float32)

    # Aplano a 2D
    X_flat = X.reshape(-1, X.shape[-1])  # [num_ejemplos*num_acciones, 4]

    # Fit/transform
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # Vuelvo a la forma original
    X = X_scaled.reshape(-1, X.shape[1], X.shape[2])
    return X

def load_dataset(dataset_file):
    X_src, X_tgt, Y, blocks_ids = load_data_from_file(dataset_file)
    X_tgt = normalize_input(X_tgt)
    X_tgt = torch.tensor(X_tgt, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    return TensorDataset(X_tgt, Y)

def create_train_subsets(train_datasets, subset_size, seed=42):
    def create_subset(datasets):
        size_per_set = subset_size // len(datasets)
        subsets = []
        for dataset in datasets:
            subset, _ = random_split(dataset, [size_per_set, len(dataset) - size_per_set])
            subsets.append(subset)
        
        # Unir todos los subsets en uno solo
        return ConcatDataset(subsets)

    torch.manual_seed(seed)
    train_subsets = []

    current_datasets = []
    for train_dataset in train_datasets:
        current_datasets.append(train_dataset)
        train_subset = create_subset(current_datasets)
        train_subsets.append(train_subset)

    return train_subsets

def create_subsets(train_dataset_files, train_size, test_dataset_files, test_size, seed=42):
    torch.manual_seed(seed)

    test_subsets = []
    train_datasets = [None for _ in range(len(train_dataset_files))]

    for test_dataset_file in test_dataset_files:
        test_dataset = load_dataset(test_dataset_file)
        test_subset, complement_set = random_split(test_dataset, [test_size, len(test_dataset) - test_size])
        test_subsets.append(NamedSubset(subset=test_subset, name=test_dataset_file))

        if test_dataset_file in train_dataset_files:
            train_datasets[train_dataset_files.index(test_dataset_file)] = complement_set

    for i, train_dataset_file in enumerate(train_dataset_files):
        if train_datasets[i] == None:
            train_datasets[i] = load_dataset(train_dataset_file)
    
    train_subsets = create_train_subsets(train_datasets, train_size)
    
    return train_subsets, test_subsets


def train(model, epochs, train_set, test_sets, batch_size, learning_rate, patience, seed=42) -> tuple[nn.Module, Metrics, list[Metrics]]:
    torch.manual_seed(seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    test_loaders = []
    for test_set in test_sets: 
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loaders.append((test_loader, test_set.name, Metrics()))

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_metrics = Metrics()

        for X_tgt_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_tgt_batch)
            loss = loss_function(outputs, y_batch.argmax(dim=-1))
            loss.backward()
            optimizer.step()

            train_metrics.update_loss(loss.item(), X_tgt_batch.size(0))
            train_metrics.update_accuracy(outputs, y_batch)

        # Validación
        model.eval()
        val_loss = 0

        for test_loader, test_set, metrics in test_loaders:
            with torch.no_grad():
                for X_tgt_batch, y_batch in test_loader:
                    outputs = model(X_tgt_batch)
                    loss = loss_function(outputs, y_batch.argmax(dim=-1))

                    metrics.update_loss(loss.item(), X_tgt_batch.size(0))
                    metrics.update_accuracy(outputs, y_batch)
                    val_loss += metrics.loss_history[-1]
            
        # Print epoch results
        print(f'Epoch {epoch + 1}/{epochs} - '
            f'Train Loss: {train_metrics.loss_history[-1]:.4f}, Train Accuracy: {train_metrics.acc_history[-1]:.2f}%'
            f' - Val Loss: {sum(m.loss for _, _, m in test_loaders) / sum(m.total for _, _, m in test_loaders):.4f}, '
            f'Val Accuracy: {np.mean([m.acc_history[-1] for _, _, m in test_loaders]):.2f}%')
        for _, test_set, m in test_loaders:
            print(f'    Test Set: {test_set} - '
                f'Val Loss: {m.loss_history[-1]:.4f}, Val Accuracy: {m.acc_history[-1]:.2f}%')
            

        # --- EARLY STOPPING SOLO EN LOSS ---
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping en la epoch {epoch+1}")
                model.load_state_dict(best_model_wts)
                return model, train_metrics, [[test_set, metrics] for _, test_set, metrics in test_loaders]

    # Al terminar todas las fases, restauramos el mejor modelo
    model.load_state_dict(best_model_wts)
    return model, train_metrics, [[test_set, metrics] for _, test_set, metrics in test_loaders]

def curriculum_learning(model, epochs, train_dataset_files, train_size, test_dataset_files, test_size, batch_size, learning_rate, patience, seed=42):
    train_sets, test_sets = create_subsets(train_dataset_files, train_size, test_dataset_files, test_size, seed)

    for train_set in train_sets:
        train(model, epochs, train_set, test_sets, batch_size, learning_rate, patience, seed)
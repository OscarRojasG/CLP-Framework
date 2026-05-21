from abc import ABC, abstractmethod
from collections import defaultdict
import torch.nn.functional as F
import torch

class Metric(ABC):
    def __init__(self, name: str, maximize: bool = True):
        self.name = name
        self.maximize = maximize
        self.reset()

    @abstractmethod
    def reset(self): pass

    @abstractmethod
    def step(self, logits, y): pass

    @abstractmethod
    def _compute(self): pass

    def compute(self):
        val = self._compute()
        self.reset()
        return val

    def format(self, value: float) -> str:
        return f"{value:.2f}"

class EpochMetrics:
    def __init__(self):
        # Estructura: subset_name -> metric -> list[values]
        self.subset_metrics = defaultdict(lambda: defaultdict(list))

    def add_value(self, subset_name: str, metric: Metric, value: float):
        self.subset_metrics[subset_name][metric].append(value)

    def get_last_metric_value(self, subset_name: str, metric: Metric):
        return self.subset_metrics[subset_name][metric][-1]

class MeanRank(Metric):
    def __init__(self):
        super().__init__("MR")

    def reset(self):
        self.total_ranks = []

    def step(self, logits, y):
        labels = y.argmax(dim=-1) 
        _, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        matched_positions = (sorted_indices == labels.unsqueeze(1)).nonzero()[:, 1]
        ranks = matched_positions + 1
        self.total_ranks.extend(ranks.cpu().tolist())

    def _compute(self):
        all_ranks_tensor = torch.tensor(self.total_ranks, dtype=torch.float)
        return all_ranks_tensor.mean().item()
    
class Accuracy(Metric):
    def __init__(self, k=1):
        super().__init__("Accuracy" if k == 1 else f"Top-{k} Accuracy")
        self.k = k

    def reset(self):
        self.total_correct = 0
        self.total_samples = 0
    
    def step(self, logits, y):
        batch_size = y.size(0)
        target_indices = y.argmax(dim=-1)
        _, top_k_indices = logits.topk(self.k, dim=1, largest=True, sorted=True)
        correct = top_k_indices.eq(target_indices.view(-1, 1).expand_as(top_k_indices))
        self.total_correct += correct.sum().item()
        self.total_samples += batch_size

    def _compute(self):
        return 100 * self.total_correct / self.total_samples
    
    def format(self, value):
        return f"{value:.2f}%"
    
class KL_Divergence(Metric):
    def __init__(self, tau):
        super().__init__("KL Div.", False)
        self.tau = tau

    def reset(self):
        self.total_samples = 0
        self.total_kl = 0
    
    def step(self, logits, y):
        y = torch.softmax(y / self.tau, dim=-1)
        log_preds = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(log_preds, y, reduction='batchmean')
        batch_size = y.size(0)
        self.total_kl += kl.item() * batch_size 
        self.total_samples += batch_size
        return kl

    def _compute(self):
        return self.total_kl / self.total_samples
    
class CrossEntropyLoss(Metric):
    def __init__(self):
        super().__init__("CrossEntropy", False)

    def reset(self):
        self.total_samples = 0
        self.total_ce = 0
    
    def step(self, logits, y):
        batch_size = y.size(0)

        y = y / y.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        ce = F.cross_entropy(logits, y, reduction='mean')

        self.total_ce += ce.item() * batch_size 
        self.total_samples += batch_size
        
        return ce

    def _compute(self):
        if self.total_samples == 0:
            return 0.0
        return self.total_ce / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"

class MeanReciprocalRank(Metric):
    def __init__(self):
        super().__init__("MRR")

    def reset(self):
        self.reciprocal_ranks = []

    def step(self, logits, y):
        labels = y.argmax(dim=-1) 
        _, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        matched_positions = (sorted_indices == labels.unsqueeze(1)).nonzero()[:, 1]
        ranks = matched_positions + 1
        reciprocals = 1.0 / ranks.to(torch.float)     
        self.reciprocal_ranks.extend(reciprocals.cpu().tolist())

    def _compute(self):
        if not self.reciprocal_ranks:
            return 0.0
        
        all_reciprocals_tensor = torch.tensor(self.reciprocal_ranks, dtype=torch.float)
        return all_reciprocals_tensor.mean().item()
    
    def format(self, value):
        return f"{value:.3f}"
    
class MSE(Metric):
    def __init__(self):
        super().__init__("MSE", False)

    def reset(self):
        self.total_samples = 0
        self.total_mse = 0
    
    def step(self, logits, y):
        mse = torch.nn.functional.mse_loss(logits, y.float())
        
        batch_size = y.size(0)
        self.total_mse += mse.item() * batch_size
        self.total_samples += batch_size
        
        return mse

    def _compute(self):
        if self.total_samples == 0: return 0.0
        return self.total_mse / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"
    
class ExpMSE(Metric):
    def __init__(self):
        super().__init__("ExpMSE", False)

    def reset(self):
        self.total_samples = 0
        self.total_mse_real = 0
    
    def step(self, logits, y_log):
        """
        logits: Salida del modelo (en escala logarítmica)
        y_log: Target original (en escala logarítmica)
        """
        # 1. Revertimos la transformación log para ambos
        preds_real = torch.exp(logits)
        targets_real = torch.exp(y_log)
        
        # 2. Calculamos el MSE en la escala original de pasos/desperdicio
        mse_real = torch.nn.functional.mse_loss(preds_real, targets_real.float())
        
        # 3. Acumulamos usando el batch size
        batch_size = y_log.size(0)
        self.total_mse_real += mse_real.item() * batch_size
        self.total_samples += batch_size
        
        return mse_real

    def _compute(self):
        if self.total_samples == 0: 
            return 0.0
        return self.total_mse_real / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"
    
class MAE(Metric):
    def __init__(self):
        super().__init__("MAE", False)

    def reset(self):
        self.total_samples = 0
        self.total_mse = 0
    
    def step(self, logits, y):
        mse = torch.nn.functional.l1_loss(logits, y.float())
        
        batch_size = y.size(0)
        self.total_mse += mse.item() * batch_size
        self.total_samples += batch_size
        
        return mse

    def _compute(self):
        if self.total_samples == 0: return 0.0
        return self.total_mse / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"
    
class ExpMAE(Metric):
    def __init__(self):
        super().__init__("ExpMAE", False)

    def reset(self):
        self.total_samples = 0
        self.total_mse_real = 0
    
    def step(self, logits, y_log):
        """
        logits: Salida del modelo (en escala logarítmica)
        y_log: Target original (en escala logarítmica)
        """
        # 1. Revertimos la transformación log para ambos
        preds_real = torch.exp(logits)
        targets_real = torch.exp(y_log)
        
        # 2. Calculamos el MSE en la escala original de pasos/desperdicio
        mse_real = torch.nn.functional.l1_loss(preds_real, targets_real.float())
        
        # 3. Acumulamos usando el batch size
        batch_size = y_log.size(0)
        self.total_mse_real += mse_real.item() * batch_size
        self.total_samples += batch_size
        
        return mse_real

    def _compute(self):
        if self.total_samples == 0: 
            return 0.0
        return self.total_mse_real / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"
    
class ValidActionAccuracy(Metric):
    def __init__(self):
        # Nombramos la métrica para identificarla en los logs de entrenamiento/validación
        super().__init__("Valid Action Accuracy")

    def reset(self):
        self.total_correct = 0
        self.total_samples = 0
    
    def step(self, logits, y):
        # logits: [Batch_Size, Max_Actions] -> Predicciones crudas del modelo
        # y: [Batch_Size, Max_Actions] -> Distribución de probabilidades (targets suaves)
        batch_size = y.size(0)
        
        # 1. Obtenemos el índice de la acción que el modelo decidió escoger (el valor más alto)
        model_choices = logits.argmax(dim=-1) # [Batch_Size]
        
        # 2. Extraemos el valor que tenía el target real en las posiciones elegidas por el modelo
        # Arange genera los índices de las filas del batch para extraer el elemento correcto por cada muestra
        row_indices = torch.arange(batch_size, device=y.device)
        chosen_target_values = y[row_indices, model_choices] # [Batch_Size]
        
        # 3. Una muestra es correcta si el valor en el target real es estrictamente mayor que cero
        correct = chosen_target_values > 0.0 # Tensor booleano [Batch_Size]
        
        # 4. Acumulamos los resultados del batch actual
        self.total_correct += correct.sum().item()
        self.total_samples += batch_size

    def _compute(self):
        if self.total_samples == 0:
            return 0.0
        return 100 * self.total_correct / self.total_samples
    
    def format(self, value):
        return f"{value:.2f}%"
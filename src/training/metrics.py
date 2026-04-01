from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch

class EpochMetrics():
    def __init__(self):
        self.subset_metrics = {}

    def add_value(self, subset_name, metric_cls, value):
        if subset_name not in self.subset_metrics:
            self.subset_metrics[subset_name] = {}
        
        if metric_cls not in self.subset_metrics[subset_name]:
            self.subset_metrics[subset_name][metric_cls] = []
            
        self.subset_metrics[subset_name][metric_cls].append(value)

class Metric(ABC):
    def __init__(self, name, maximize=True):
        self.name = name
        self.maximize = maximize
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, logits, y):
        pass

    @abstractmethod
    def _compute(self):
        pass

    def compute(self):
        value = self._compute()
        self.reset()
        return value

    def format(self, value):
        return f"{value:.2f}"

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
        ce = torch.nn.functional.cross_entropy(logits, y)
        batch_size = y.size(0)
        self.total_ce += ce.item() * batch_size 
        self.total_samples += batch_size
        return ce

    def _compute(self):
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
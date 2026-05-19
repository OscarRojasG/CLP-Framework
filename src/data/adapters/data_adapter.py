import numpy as np
from abc import ABC, abstractmethod

class DataAdapter(ABC):
    def __init__(self, keys):
        super().__init__()
        self.data = {
            k: [] for k in keys
        }
        self.keys = keys

    def add(self, sample: tuple):
        for k, v in zip(self.keys, sample):
            self.data[k].append(v)

    def get(self) -> dict:
        return {
            k: np.stack(v, dtype=self.keys[k]) for k, v in self.data.items()
        }

    def count(self):
        return len(self.data[list(self.data.keys())[0]])
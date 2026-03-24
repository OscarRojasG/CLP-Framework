from abc import ABC, abstractmethod

class GreedySolver(ABC):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def solve(self, instance_file, instance_number, min_fr):
        pass
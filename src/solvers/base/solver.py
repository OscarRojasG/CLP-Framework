from abc import ABC, abstractmethod

class Solver(ABC):
    @abstractmethod
    def solve(self, instance_file, instance_number, w: int) -> int:
        pass
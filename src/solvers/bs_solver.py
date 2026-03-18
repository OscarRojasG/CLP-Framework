from abc import ABC, abstractmethod

class BS_Solver(ABC):
    @abstractmethod
    def solve(self, instance_file, instance_number) -> int:
        pass
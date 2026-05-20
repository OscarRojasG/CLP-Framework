from abc import ABC, abstractmethod
from settings import INSTANCE_FOLDER

class Solver(ABC):
    def __init__(self, name, min_fr):
        self.name = name
        self.min_fr = min_fr

    def solve_all(self, file):
        path = INSTANCE_FOLDER / file
        with open(path, 'r') as f:
            num_instances = int(f.readline())

        vols = []
        times = []
        for i in range(num_instances):
            vol, time = self.solve(file, i)
            vols.append(vol)
            times.append(time)

        return vols, times

    @abstractmethod
    def solve(self, file, instance_number):
        pass

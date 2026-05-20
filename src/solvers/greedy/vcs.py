import subprocess
from settings import VCS_SOLVER_PATH, INSTANCE_FOLDER
from solvers.solver import Solver

class VCSSolver(Solver):
    def __init__(self, min_fr):
        super().__init__("VCS", min_fr)

    def solve(self, instance_file, instance_number):
        # Ejecutar el proceso y capturar la salida
        proc = subprocess.run(
            [VCS_SOLVER_PATH, INSTANCE_FOLDER / instance_file, "-i", str(instance_number), f"--min_fr={self.min_fr}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )

        lines = proc.stdout.strip().splitlines()
        volume = float(lines[-3])
        time = float(lines[-1])
        return volume, time
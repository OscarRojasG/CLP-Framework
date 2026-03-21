import subprocess
from settings import VCS_SOLVER_PATH, INSTANCE_FOLDER
from solvers.greedy.greedy_solver import GreedySolver

class VCSSolver(GreedySolver):
    def __init__(self):
        super().__init__("VCS")

    def solve(self, instance_file, instance_number, min_fr) -> int:
        # Ejecutar el proceso y capturar la salida
        proc = subprocess.run(
            [VCS_SOLVER_PATH, INSTANCE_FOLDER / instance_file, "-i", str(instance_number), f"--min_fr={min_fr}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )

        line = proc.stdout.strip().splitlines()[-1]
        volume = float(line)
        return volume
from solvers.base.solver import Solver
import subprocess
from settings import BSG_SOLVER_PATH, INSTANCE_FOLDER

class BSGSolver(Solver):
    def solve(self, instance_file, instance_number, w: int) -> int:
        # Ejecutar el proceso y capturar la salida
        proc = subprocess.run(
            [BSG_SOLVER_PATH, INSTANCE_FOLDER / instance_file, "-i", str(instance_number), "-w", str(w)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )

        line = proc.stdout.strip().splitlines()[-1]
        volume = float(line)
        return volume
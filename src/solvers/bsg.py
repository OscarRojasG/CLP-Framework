import subprocess
from settings import BSG_SOLVER_PATH, INSTANCE_FOLDER
from solvers.bs_solver import BS_Solver

class BSG_Solver(BS_Solver):
    def __init__(self, w):
        self.w = w
        
    def solve(self, instance_file, instance_number) -> int:
        # Ejecutar el proceso y capturar la salida
        proc = subprocess.run(
            [BSG_SOLVER_PATH, INSTANCE_FOLDER / instance_file, "-i", str(instance_number), "-w", str(self.w)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )

        line = proc.stdout.strip().splitlines()[-1]
        volume = float(line)
        return volume
import subprocess
from settings import BSG_SOLVER_PATH, INSTANCE_FOLDER
from solvers.timed.timed_solver import Timed_Solver

class Timed_BSG_Solver(Timed_Solver):
    def __init__(self):
        super().__init__("Timed BSG")
        
    def solve(self, instance_file, instance_number, min_fr, time):
        # Ejecutar el proceso y capturar la salida
        proc = subprocess.run(
            [BSG_SOLVER_PATH, INSTANCE_FOLDER / instance_file, "-i", str(instance_number), "-t", str(time), f"--min_fr={min_fr}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )

        line = proc.stdout.strip().splitlines()[-3]
        volume = float(line)
        return volume
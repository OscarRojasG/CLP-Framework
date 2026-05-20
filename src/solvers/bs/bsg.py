import subprocess
from settings import BSG_SOLVER_PATH, INSTANCE_FOLDER
from solvers.bs.bsm_solver import BS_Solver

class BSG_Solver(BS_Solver):
    def __init__(self):
        super().__init__("BSG")
        
    def solve(self, instance_file, instance_number, w, min_fr) -> int:
        # Ejecutar el proceso y capturar la salida
        proc = subprocess.run(
            [BSG_SOLVER_PATH, INSTANCE_FOLDER / instance_file, "-i", str(instance_number), "-w", str(w), f"--min_fr={min_fr}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )

        lines = proc.stdout.strip().splitlines()
        volume = float(lines[-3])
        time = float(lines[-1])
        return volume, time
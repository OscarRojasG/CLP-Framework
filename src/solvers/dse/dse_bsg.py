import subprocess
from settings import BSG_SOLVER_PATH, INSTANCE_FOLDER
from solvers.dse.dse_solver import DSE_Solver

class DSE_BSG_Solver(DSE_Solver):
    def __init__(self):
        super().__init__("DSE BSG")
        
    def solve(self, instance_file, instance_number, min_fr, max_w) -> int:
        # Ejecutar el proceso y capturar la salida
        proc = subprocess.run(
            [BSG_SOLVER_PATH, INSTANCE_FOLDER / instance_file, "-i", str(instance_number), "-w", str(max_w), f"--min_fr={min_fr}", "--de"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )

        lines = proc.stdout.strip().splitlines()
        volume = float(lines[-3])
        time = float(lines[-1])
        return volume, time
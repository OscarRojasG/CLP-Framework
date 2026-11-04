import subprocess
from settings import BSG_SOLVER_PATH, BSG_FIXED_W_SOLVER_PATH, INSTANCE_FOLDER
from solvers.solver import Solver

class BSG_VCS_Solver(Solver):
    def __init__(self, w, min_fr, fixed_w=False):
        super().__init__("BSG", min_fr)
        self.w = w
        self.solver_path = BSG_FIXED_W_SOLVER_PATH if fixed_w else BSG_SOLVER_PATH

    def solve(self, instance_file, instance_number) -> int:
        # Ejecutar el proceso y capturar la salida
        proc = subprocess.run(
            [self.solver_path, INSTANCE_FOLDER / instance_file, "-i", str(instance_number), "-w", str(self.w), f"--min_fr={self.min_fr}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )

        lines = proc.stdout.strip().splitlines()
        volume = float(lines[-3])
        time = float(lines[-1])
        return volume, time
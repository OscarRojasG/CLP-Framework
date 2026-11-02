import subprocess
from settings import BSG_SOLVER_PATH, INSTANCE_FOLDER
from solvers.solver import Solver

class DE_BSG_VCS_Solver(Solver):
    def __init__(self, timelimit, min_fr):
        super().__init__("DE-BSG", min_fr)
        self.timelimit = timelimit

    def solve(self, instance_file, instance_number) -> int:
        # Con -t, BSG_CLP ejecuta DoubleEffort (w inicial 4) hasta alcanzar el tiempo límite
        proc = subprocess.run(
            [BSG_SOLVER_PATH, INSTANCE_FOLDER / instance_file, "-i", str(instance_number), "-t", str(self.timelimit), f"--min_fr={self.min_fr}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )

        lines = proc.stdout.strip().splitlines()
        volume = float(lines[-3])
        time = float(lines[-1])
        return volume, time

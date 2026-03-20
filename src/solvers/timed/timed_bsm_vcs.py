from solvers.bs.bsm_vcs import BSM_VCS_Solver
from models.base.transformer import Transformer
from solvers.timed.timed_solver import Timed_BSM_Solver
from bsm_engine import DoubleEffort_BSM_VCS

class Timed_BSM_VCS_Solver(Timed_BSM_Solver):
    def __init__(self, model: Transformer):
        super().__init__(model, "Timed BSM-VCS", DoubleEffort_BSM_VCS, BSM_VCS_Solver)
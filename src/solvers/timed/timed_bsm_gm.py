from solvers.bs.bsm_gm import BSM_GM_Solver
from models.base.transformer import Transformer
from solvers.timed.timed_solver import Timed_BSM_Solver
from bsm_engine import DoubleEffort_BSM_GM

class Timed_BSM_GM_Solver(Timed_BSM_Solver):
    def __init__(self, model: Transformer):
        super().__init__(model, "Timed BSM-GM", DoubleEffort_BSM_GM, BSM_GM_Solver)
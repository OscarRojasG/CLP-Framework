from solvers.bs.bsm_gm import BSM_GM_Solver
from models.base.transformer import Transformer
from solvers.dse.dse_solver import DSE_BSM_Solver
from bsm_engine import DoubleEffort_BSM_GM

class DSE_BSM_GM_Solver(DSE_BSM_Solver):
    def __init__(self, model: Transformer):
        super().__init__(model, "DSE BSM-GM", DoubleEffort_BSM_GM, BSM_GM_Solver)
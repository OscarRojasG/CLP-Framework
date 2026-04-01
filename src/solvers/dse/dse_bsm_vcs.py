from solvers.bs.bsm_vcs import BSM_VCS_Solver
from models.base.transformer import Transformer
from solvers.dse.dse_solver import DSE_BSM_Solver
from bsm_engine import DoubleEffort_BSM_VCS

class DSE_BSM_VCS_Solver(DSE_BSM_Solver):
    def __init__(self, model: Transformer):
        super().__init__(model, "DSE BSM-VCS", DoubleEffort_BSM_VCS, BSM_VCS_Solver)
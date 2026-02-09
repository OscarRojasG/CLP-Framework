from solvers.bsm import BSMSolver
from models.base.transformer import Transformer
import math
import signal

class TimeoutException(Exception):
    pass

class TimedBSMSolver():
    def __init__(self, model: Transformer):
        self.model = model
        
    def solve(self, instance_file, instance_number, time):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(time)

        bsm_solver = BSMSolver(self.model)
        w = 1
        best_eval = 0
        bsm = None

        try:
            while True:
                bsm = bsm_solver.env.init(instance_file, instance_number, w)
                bsm_solver._solve(w, bsm)
                best_eval = max(best_eval, bsm.get_volume_ratio() * 100)
                w = math.ceil(w * math.sqrt(2))
        except TimeoutException:
            pass
        finally:
            signal.alarm(0)

        if bsm is not None:
            best_eval = max(best_eval, bsm.get_volume_ratio() * 100)
            bsm.close()
        return best_eval

    def timeout_handler(self, signum, frame):
        raise TimeoutException()
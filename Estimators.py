import numpy as np
import matplotlib.pyplot as plt
import Models as mod
import Distribution as dist
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")

class Estimator:
    def __init__(self, model):
        self.model = model
        
    def get_estimator(self, ts:np.ndarray):
        pass
    
class MLE(Estimator):
    def __init__(self, model):
        super().__init__(model)
        
    def get_estimator(self, ts:np.ndarray, idx_params:np.ndarray, init_guess=None):
        
        x0 = init_guess
        constr = self.model.get_constraints()
        bounds = self.model.get_variable_bounds()
        
        d = dist.Normal()
        d.add_model(self.model)
        
        res = minimize(d.cond_log_likelihood, x0, constraints=constr, bounds=bounds, args=(ts, idx_params))
        
        return [res.x[idx_params[i]] for i in range(len(idx_params))], res
    
    
    
    
    
    
    
    
    

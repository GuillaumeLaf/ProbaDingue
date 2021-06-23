import numpy as np
import matplotlib.pyplot as plt
import Models as mod
import Distribution as dist
from scipy.optimize import minimize
from Arrays import *

import warnings
warnings.filterwarnings("ignore")

class Estimator:
    def __init__(self, model):
        self.model = model
        
    def get_estimator(self, ts:Array):
        pass
    
class MLE(Estimator):
    def __init__(self, model):
        super().__init__(model)
        
    def get_estimator(self, ts:Array, idx_params:np.ndarray, init_guess=None):
        
        x0 = init_guess
        constr = self.model.get_constraints()
        bounds = self.model.get_variable_bounds()
        
        d = dist.Normal()   # We should be able to specify other distributions
        d.add_model(self.model)
              
        
        res = minimize(d.neg_log_likelihood, x0, constraints=constr, bounds=bounds, args=(ts(), idx_params), method='SLSQP')
        
        estimators_var = np.linalg.inv(d.hessian_log_likelihood(res.x, ts(), idx_params))
        estimators_var *= -1.0
        estimators_var = np.squeeze(estimators_var)
        estimators_var = np.diag(estimators_var)
        
        return [res.x[idx_params[i]] for i in range(len(idx_params))], [estimators_var[idx_params[i]] for i in range(len(idx_params))], res
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    

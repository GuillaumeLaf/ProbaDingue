import numpy as np
import matplotlib.pyplot as plt
import Models as mod
from scipy.optimize import minimize

class Estimator:
    def __init__(self, model):
        self.model = model
        
    def get_estimator(self, ts:np.ndarray):
        pass
    
class MLE(Estimator):
    def __init__(self, model):
        super().__init__(model)
        
    def get_estimator(self, ts:np.ndarray, idx_params:np.ndarray):
        
        x0 = self.model.get_initial_guess()
        constr = self.model.get_constraints()
        
        res = minimize(self.normal_likelihood, x0, constraints=constr, args=(ts, idx_params))
        
        return [res.x[idx_params[i]] for i in range(idx_params.shape[0])], res
        
    def normal_likelihood(self, params:np.ndarray, x:np.ndarray, idx_params:np.ndarray):
        cond_var = self.model.get_conditional_variance(params[idx_params[0]])
        cond_exp = self.model.get_conditional_expectation(params[idx_params[1]])
        
        ssq = np.sum((x - cond_exp)**2.0)
        likelihood = (len(x) * np.log(2.0*np.pi*cond_var))/2.0 + ssq/(2.0 * cond_var)
        return likelihood / 100000.0
    

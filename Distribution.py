import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
# from Models import *
import autograd.numpy as anp
from autograd import grad, jacobian, hessian

class Distribution:
    def __init__(self):
        """
        Interface for the probability distributions

        Returns
        -------
        None.

        """
        
        pass
    
    def add_model(self, model):
        self.model = model
        
    def grad_log_likelihood(self, params:np.ndarray, x:np.ndarray, idx_params:np.ndarray):
        return grad(self.neg_log_likelihood, argnum=0)(params, x, idx_params)
    
    def hessian_log_likelihood(self, params:np.ndarray, x:np.ndarray, idx_params:np.ndarray):
        return hessian(self.log_likelihood, argnum=0)(params, x, idx_params)
    
class Normal(Distribution):
    def __init__(self):
        pass
    
    def log_likelihood(self, params:np.ndarray, x:np.ndarray, idx_params:np.ndarray):
        mu = self.model.get_conditional_expectation(params[idx_params[1]])
        var = self.model.get_conditional_variance(params[idx_params[0]])
        ssq = anp.sum((x - mu)**2.0)
        return -(len(x) * anp.log(2.0*anp.pi*var))/2.0 - ssq/(2.0*var)
    
    def neg_log_likelihood(self, params:np.ndarray, x:np.ndarray, idx_params:np.ndarray):
        return -1.0 * self.log_likelihood(params, x, idx_params) / (x.size*10.0)
    
    
  

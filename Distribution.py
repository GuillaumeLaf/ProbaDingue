import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
# from Models import mod

class Distribution:
    def __init__(self):
        pass
    
    def add_model(self, model):
        self.model = model
    
class Normal(Distribution):
    def __init__(self):
        pass
    
    def cond_log_likelihood(self, params:np.ndarray, x:np.ndarray, idx_params:np.ndarray):
        # This function will be passed to the 'minimize' function
        cond_exp = self.model.get_conditional_expectation(params[idx_params[1]])
        cond_var = self.model.get_conditional_variance(params[idx_params[0]])
        
        return self.log_likelihood(cond_exp, cond_var, x)
    
    # @staticmethod
    # @nb.njit()
    def log_likelihood(self, mu:np.ndarray, var:np.ndarray, x:np.ndarray):
        ssq = np.sum((x - mu)**2.0)
        return (len(x) * np.log(var))/2.0 + ssq/(2.0*var)
        
    
    
    
    
    
    
    
    
        

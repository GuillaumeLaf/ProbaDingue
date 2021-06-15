import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
# from Models import *

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
        
    
class Normal(Distribution):
    def __init__(self):
        pass
    
    def cond_log_likelihood(self, params:np.ndarray, x:np.ndarray, idx_params:np.ndarray):
        # This function will be passed to the 'minimize' function
        cond_exp = self.model.get_conditional_expectation(params[idx_params[1]])
        cond_var = self.model.get_conditional_variance(params[idx_params[0]])
        
        return self.log_likelihood(cond_var, cond_exp, x)
    
    def log_likelihood(self, var:np.float64, mu:np.ndarray, x:np.ndarray):
        ssq = np.sum((x - mu)**2.0)
        return (len(x) * np.log(var))/2.0 + ssq/(2.0*var)
    
    # Use chain rule to get the Jacobian and Hessian.
    
    def grad_log_likelihood(self):
        pass
    
    def grad_mu(self):  # need to take the gradient since it is a function of the params.
        pass
    
    def grad_var(self):
        pass
    
    def total_gradiant(self):
        pass
        

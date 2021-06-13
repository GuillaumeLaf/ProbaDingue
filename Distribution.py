import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import autograd.numpy as anp
from autograd import grad
from Models import *

class Distribution:
    model:Model
    def __init__(self):
        pass
    
    def add_model(self, model:Model):
        self.model = model
     
    def grad_log_likelihood(self, *args):
        return np.array([grad(self.log_likelihood, argnum=0)(*args)])
    
class Normal(Distribution):
    def __init__(self):
        pass
    
    def log_likelihood(self, *args):
        # This function should accept arrays
        # Returns the sum of the likelihoods for each elements
        # Should be able to be differentiable with autograd
        mu = args[0]
        var = args[1]
        x = args[2]
        
        ssq = anp.sum((x - mu)**2.0)
        return (len(x) * anp.log(var))/2.0 + ssq/(2.0*var)
    
    
        

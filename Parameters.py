import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Parameters:
    def __init__(self):
        pass
    
class AR_parameters(Parameters):
    phis:np.ndarray
    phis_var:np.ndarray
    var_e:np.float64
    var_e_var:np.float64
    def __init__(self, order:np.int8):
        self.order = order
        
    def set_phis(self, phis:np.ndarray):
        self.phis = phis
        
    def set_var_e(self, var_e:np.float64):
        self.var_e = var_e
        
    def set_phis_var(self, phis_var:np.ndarray):
        self.phis_var = phis_var
        
    def set_var_e_var(self, var_e_var:np.ndarray):
        self.var_e_var = var_e_var
        
    def to_dict(self):
        d = {'order':self.order, 
             'var_e':self.var_e[0]}
        for i in range(len(self.phis)):
            d['phi_'+str(i+1)] = self.phis[i]
        return d
    
    def to_array(self):
        return np.concatenate((self.var_e, self.phis))






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        pass
    
    def fit(self):
        pass
    
    def sample(self, n:np.int64):
        pass
    

class AR(Model):
    order:np.int8
    phis:np.ndarray
    sigma_e:np.float64
    def __init__(self, order:np.int8):
        self.order = order
    
    def draw(self):
        # Need to check if the parameters are initialised !
        # The first element of 'prev' is the most recent obs
        prev = np.random.normal(0,1, (self.order))
        
        while True:
            current_innov = np.random.normal(0,1)
            current = self.phis @ prev + current_innov
            yield current
            prev = np.concatentate((current, prev[:-1]))
            


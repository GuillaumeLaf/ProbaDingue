import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Transformations import *

class Array:
    def __init__(self, array:np.ndarray):
        self.array = array
        
    def __call__(self):
        return self.array
    
    def __getitem__(self, key):
        return self.array[key]
    
    def __setitem__(self, key, value):
        self.array[key] = value
        
    def __getslice__(self, i, j):
        return self.array[i:j]
    
    def __setslice__(self, i, j, s):
        self.array[i:j] = s

class TransformedTS(Array):
    transformations:dict
    def __init__(self, array:np.ndarray):
        super().__init__(array)
    
class LevelTS(Array):
    def __init__(self, array:np.ndarray):
        super().__init__(array)
    
class Transform_Pipeline:
    transformations:dict
    def __init__(self):
        pass
    
    def __add__(self, other):
        pass
    
    def __iadd__(self, other):
        pass
    
    def __radd__(self, other):
        pass
    
    # Use a context manager to set-up the pipeline.
    # Be able to __add__ different Transformation Object ?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Array:
    def __init__(self, array:np.ndarray):
        self.array = array
        
    def __call__(self):
        return self.array
    

class TransformedTS(Array):
    transformations:dict
    def __init__(self, array:np.ndarray):
        super().__init__(array)
    
class LevelTS(Array):
    def __init__(self, array:np.ndarray):
        super().__init__(array)
    
class Transform_Pipeline:
    def __init__(self):
        pass
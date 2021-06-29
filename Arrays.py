import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Transformations import *
from copy import deepcopy

class Array:
    def __init__(self, array:np.ndarray):
        self.array = array
        
    def __call__(self):
        return self.array
    
    def __len__(self):
        return len(self.array)
    
    def __getitem__(self, key):
        return self.array[key]
    
    def __setitem__(self, key, value):
        self.array[key] = value
        
    def __getslice__(self, i, j):
        return self.array[i:j]
    
    def __setslice__(self, i, j, s):
        self.array[i:j] = s
        
    def concatenate(self, other):
        pass
    
    def apply(self, f):
        self.array = f(self.array)
        return self

class TransformedTS(Array):
    transformations:dict # will be a dict of dicts (since we have the parameters for the reverse transfo.).
    def __init__(self, array:np.ndarray):   
        # Most recent observation at the end of array !
        super().__init__(array)
    
    def concatenate(self, other):
        # Check if 'self' and 'other' have the same transfo.
        if isinstance(other, TransformedTS):
            concat_array = np.concatenate((self.array, other.array))
            tmp = TransformedTS(concat_array)
            tmp.transformations = self.transformations
            return tmp  # Return a new object
        else:
            raise ValueError("Must concatenate between two TransformedTS.")
        
class TS(Array):
    def __init__(self, array:np.ndarray):
        # Most recent observation at the end of array !
        super().__init__(array)
        
    def set_name(self, name:str):
        self.name = name
    

class Transform_Pipeline:
    # This class will use a context manager to add and apply the transforms to the TS.
    # The context manager will extract the array from TS object, hence inside the context we are working with ndarray.
    def __init__(self):
        self.transformations = {}
    
    def __add__(self, other):
        # use to add some transformation to the dict. 
        # Note that the transfo. object must already be initialised.
        if isinstance(other, Transformation):
            self.transformations[other.name] = other
        else:
            raise ValueError("Cannot sum a Pipeline and an object different from Transformation")
        return self
    
    def __iadd__(self, other):
        if isinstance(other, Transformation):
            self.transformations[other.name] = other
        else:
            raise ValueError("Cannot sum a Pipeline and an object different from Transformation")
        return self
    
    def transform(self, ts:TS):
        # Run the 'ts' through the pipe. 
        # Must return a 'TransformedTS' object along with the recipe of transformations (with used params as dict).
        # the transformation params will be used to inverse the transform.
        tmp = deepcopy(ts)
        recipe = {}
        for key, value in self.transformations.items():
            tmp, recipe[key] = value.apply_transform(tmp)
            
        tmp = TransformedTS(tmp)
        tmp.transformations = recipe
            
        return tmp
    
    def reverse_transform(self, ts:TransformedTS):
        # Note that the 'TransformedTS' contains the recipe.
        tmp = deepcopy(ts)
        for key, value in reversed(ts.transformations.items()):
            tmp = self.transformations[key].reverse_transform(tmp, value)
            
        return TS(tmp)
        
        
        
        
        
        
        
        
        
        
        
        
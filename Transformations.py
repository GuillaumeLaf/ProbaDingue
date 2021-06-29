import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Transformation:
    args:dict
    def __init__(self):
        pass
    def apply_transform(self, ts:np.ndarray):
        # Must return the transformed 'ts' array (as a ndarray).
        pass

    def reverse_transform(self, ts:np.ndarray, args:dict):
        # Must return the reverse transformed 'ts' array (as a ndarray).
        pass
    
class Logarize(Transformation):
    # Take the logarithm of the TS.
    # Check if no elements are zero or negative.
    def __init__(self):
        self.name = 'Logarize'
        self.reversible = True
        self.args = {}
    
    def apply_transform(self, ts:np.ndarray):
        return np.log(ts), self.args
    
    def reverse_transform(self, ts:np.ndarray, args:dict):
        return np.exp(ts)
    
class Difference(Transformation):
    # Take the fractional difference of the serie. (The full diff. is thus a special case)
    # Note that this reduce the length of 'ts' by 1.
    def __init__(self, mode:str='full'):
        self.name = 'Difference'
        self.mode = mode    # Either 'full' for 1st diff. or 'frac' for fractional diff.
        self.reversible = True
        self.args = {'mode': mode}
    
    def apply_transform(self, ts:np.ndarray):
        coef = self.__mode_coefficient()
        self.args['coef'] = coef
        self.args['init_val'] = ts[0]
        return ts[1:] - coef * ts[:-1], self.args
    
    def reverse_transform(self, ts:np.ndarray, args:dict):
        # 'args' gives us the recipe to recover the original ts.
        if args['mode'] == 'full':
            return self.__reverseIfFull(ts, args)
        else:
            return self.__reverseIfFrac(ts, args)
            
    def __reverseIfFull(self, ts:np.ndarray, args:dict):
        rev_ts = np.concatenate((np.array([args['init_val']]), ts))
        return np.cumsum(rev_ts)
        
    def __reverseIfFrac(self, ts:np.ndarray, args:dict):
        l = len(ts)
        rev_ts = np.empty((l+1,), dtype=np.float64)
        rev_ts[0] = args['init_val']
        for i in range(l):
            rev_ts[i+1] = ts[i] + args['coef'] * rev_ts[i]
        return rev_ts
            
    def __mode_coefficient(self):
        if self.mode == 'full':
            return 1.0
        else:
            # Need to add how to compute the fractional diff. (with statistical test, ...).
            # Depends on 'ts' !!
            return 0.5  

        
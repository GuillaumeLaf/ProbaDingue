import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
import Utilities as utils
import Estimators as estim
from Parameters import *

class Model:
    def __init__(self):
        pass
    
    def get_constraints(self):
        pass
    
    def get_initial_guess(self):
        pass
    
    def fit(self):
        pass
    
    def sample(self, n:np.int64):
        sple = np.empty((n,), dtype=np.float64)
        gen = self.draw()
        for i in range(n):
            sple[i] = next(gen)
        return sple
    
class AR(Model):
    prev_x:np.ndarray
    idx_params:np.ndarray
    stable:np.bool_
    residuals:np.ndarray
    
    # Add confidence interval computation
    # Add summary statistics for model
    
    def __init__(self, order:np.int8):
        self.params = AR_parameters(order)
        
    def _check_params_initiated(self):
        if not hasattr(self.params, 'phis'):
            raise ValueError('parameters not initiated')
        
    def set_params(self, phis:np.ndarray, var_e:np.float64):
        self.params.set_phis(phis)
        self.params.set_var_e(var_e)
        self.stable = self.check_stability()
    
    @staticmethod
    @nb.njit()
    def poly_roots(phis:np.ndarray):
        """
        

        Parameters
        ----------
        phis : np.ndarray
            Parameters of the lags of the variable.

        Returns
        -------
        roots : TYPE
            array with the roots of the "inverse" polynomial.

        """
        const_term = np.array([1.0], dtype=np.float64)
        poly = np.concatenate((const_term, -phis))
        poly = np.flip(poly)
        
        roots = np.roots(poly.astype(np.complex128))
        return roots
    
    def check_stability(self):
        """
        
        Returns
        -------
        Bool
            Return True if the AR model is stable - i.e. all roots of the "inverse" polynomial are outside the unit circle.

        """
        
        return (np.abs(self.poly_roots(self.params.phis)) > 1.0).all()
    
    def draw(self):
        """
        Draw an observation from a time serie following the AR model with the given order and parameters.

        Yields
        ------
        current : TYPE
            One realization of the AR model.

        """
        
        self._check_params_initiated()
        # The first element of 'prev' is the most recent obs
        prev = np.random.normal(0,self.params.var_e, (self.params.order))
        
        while True:
            current_innov = np.random.normal(0,self.params.var_e)
            current = self.params.phis @ prev + current_innov
            yield current
            prev = np.concatenate((np.array([current]), prev[:-1]))
            
    def get_conditional_expectation(self, params:np.ndarray):
        """
        Get the conditional expectation of the model.
        Note that we need to have initiated the variable 'prev_x' 
        which gives us the observation on which we condition the expectation.
        See the function 'fit' for the initialization of the 'prev_x' array.

        Parameters
        ----------
        params : np.ndarray
            parameters of the model.

        Returns
        -------
        TYPE
            conditional expectation given 'prev_x'.

        """
        
        return np.dot(params, self.prev_x)
    
    # @staticmethod
    # @nb.njit()
    def get_conditional_variance(self, var_e:np.ndarray):
        """
        Get the conditional variance of the error terms.

        Parameters
        ----------
        var_e : np.ndarray
            variance of the error term.

        Returns
        -------
        var_e : TYPE
            variance of the error term.

        """
        
        return var_e
    
    def get_initial_guess(self):
        """
        Initial guess used for the maximization of the Likelihood.

        Returns
        -------
        nd.ndarray
            This function must return a unique array filled with the parameters.
            This array must respect the order of the parameters in the 'idx_params' array.

        """
        # Check if the inital guess is feasible
        return np.concatenate((np.array([1.0]), np.repeat(1.0/self.params.order, self.params.order)))
    
    # @staticmethod
    # @nb.njit()
    def constr_roots(self, params:np.ndarray):
        """
        Return the constraint on the 'phis' parameters of the model in order to be stable.
        This function is used for the maximization of the likelihood.

        Parameters
        ----------
        params : np.ndarray
            Parameters of the model.
            This array must respect the order of the 'idx_params' array.

        Returns
        -------
        TYPE
            ...

        """
        
        roots = self.poly_roots(params[1:].astype(np.complex128))
        roots = np.abs(roots)
        return np.min(roots) - 1.000001 # '.0000001' since inequalities are taken as non-negative
    
    def get_variable_bounds(self):
        bounds = [(0.000001, None)] # This bounds the variance to be strictly positive
        for i in range(self.params.order):
            bounds.append((None, None)) # The other parameters are not bounded
            
        return bounds
            
    
    def get_constraints(self):
        """
        Pack every constraints of the model in a unique function
        for the 'minimize' function in the 'MLE' object.

        Returns
        -------
        constr : tuple
            Tuple with the dictionaries for the constraints.

        """
        
        constr = ({'type':'ineq', 'fun': self.constr_roots})
        return constr
    
    def get_residuals(self, ts:np.ndarray):
        """
        Get the residuals after estimation for the model.
        Note that the model needs to have the parameters initiated.

        Parameters
        ----------
        ts : np.ndarray
            Initial time serie.

        Returns
        -------
            Residuals after estimation

        """
        
        self._check_params_initiated()

        resid = ts[self.params.order:] - self.params.phis @ self.prev_x
        self.residuals = np.concatenate((np.zeros((self.params.order,), dtype=np.float64), resid))   
            
    def fit(self, ts:np.ndarray, init_guess=None):
        """
        Function that fits the model with the provided time serie.
        Estimation use the MLE as estimator.

        Parameters
        ----------
        ts : np.ndarray
            Inital time serie on which the model is fitted.

        Returns
        -------
            ...

        """
        
        self.prev_x = np.array([ts[self.params.order-i:-i] for i in range(1,self.params.order+1)])
        x0 = self.get_initial_guess() if init_guess is None else init_guess
        # var_idx = np.array([0], dtype=np.int8)
        # mean_idx = np.arange(1, self.params.order+1, dtype=np.int8)
        var_idx = [0]
        mean_idx = [i for i in range(1, self.params.order+1)]
        self.idx_params = [var_idx, mean_idx]
        
        t, res = estim.MLE(self).get_estimator(ts[self.params.order:], self.idx_params, x0)
        temp_var_e, temp_phis = t
        self.params.set_phis(temp_phis)
        self.params.set_var_e(temp_var_e)
        
        self.get_residuals(ts)
        
        del(self.prev_x)
        del(self.idx_params)
    
    def predict(self, prev_x:np.ndarray, steps:np.int64):
        # prev_x must be at least the size of the order of the model
        pass

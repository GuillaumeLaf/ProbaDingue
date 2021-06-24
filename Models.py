import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
import Utilities as utils
import Estimators as estim
from Parameters import *
from functools import lru_cache
from Distribution import *
from Arrays import *
import autograd.numpy as anp

class Model:
    ts:TransformedTS
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
        sple = TransformedTS(sple)
        gen = self.draw()
        for i in range(n):
            sple[i] = next(gen)
        return sple
    
    def plot_prediction(self, n_prev:np.int16, n_predict:np.int16):
        n_total = n_predict + n_prev
        
        mean_pred = self.rolling_pred(self.end_ts(), n_predict)
        # print(mean_pred)
        std_pred = np.sqrt(self.rolling_var_pred(n_predict))
        
        mean_serie = np.concatenate((self.ts[-n_prev:], mean_pred))
        
        std_serie = np.concatenate((np.repeat(0.0, n_prev), std_pred))
        std_serie_upper = mean_serie + 1.96*std_serie
        std_serie_lower = mean_serie - 1.96*std_serie
        
        fig, ax = plt.subplots(1,1,figsize=(12, 8))
        ax.plot(self.ts[-n_prev:])
        ax.plot(mean_serie)
        ax.plot(std_serie_upper)
        ax.plot(std_serie_lower)
        
        
    
class AR(Model):
    prev_x:TransformedTS
    idx_params:np.ndarray
    stable:np.bool_
    residuals:TransformedTS
    end_ts:TransformedTS   # This variable is used to store the last obs from the ts in order to do predictions.
    # Most recent obs at the beginning
    dist:Distribution   # Have to be initialised !!!
    
    # Add confidence interval computation
    # Add summary statistics for model
    
    def __init__(self, order:np.int8):
        self.params = AR_parameters(order)
        self.dist = Normal()
        
    def _check_params_initiated(self):
        if not hasattr(self.params, 'phis'):
            raise ValueError('parameters not initiated')
        
    def set_params(self, phis:np.ndarray, var_e:np.float64):
        self.params.set_phis(phis) # The coeff. for lag 1 is at the beginning of the array.
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
        prev = np.random.normal(0,np.sqrt(self.params.var_e), (self.params.order))
        
        while True:
            current_innov = np.random.normal(0,np.sqrt(self.params.var_e))
            current = self.params.phis @ prev + current_innov
            yield current
            prev = np.concatenate((np.array([current]), prev[:-1])) # Exclude the oldest obs and put the new at the beginning
            
    def get_conditional_expectation(self, phis:np.ndarray):
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
        return anp.dot(phis, self.prev_x())
    
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
        return np.concatenate((np.array([0.1]), np.repeat(0.1, self.params.order)))
    
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
        bounds = [(1e-9, None)] # This bounds the variance to be strictly positive
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
    
    def get_residuals(self):
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

        resid = self.ts[self.params.order:] - self.params.phis @ self.prev_x()
        self.residuals = np.concatenate((np.zeros((self.params.order,), dtype=np.float64), resid))
        self.residuals = TransformedTS(self.residuals)
            
    def fit(self, ts:TransformedTS, init_guess=None):
        """
        Function that fits the model with the provided time serie.
        Estimation use the MLE as estimator.

        Parameters
        ----------
        ts : np.ndarray
            Inital time serie on which the model is fitted. Most recent observation at the end of array.

        Returns
        -------
            ...

        """
        
        self.prev_x = TransformedTS([ts[self.params.order-i:-i] for i in range(1,self.params.order+1)])
        x0 = self.get_initial_guess() if init_guess is None else init_guess

        var_idx = [0]
        mean_idx = [i for i in range(1, self.params.order+1)]
        self.idx_params = [var_idx, mean_idx]
        
        esti, esti_var, res = estim.MLE(self).get_estimator(ts[self.params.order:], self.idx_params, x0)
        self.res = res
        temp_var_e, temp_phis = esti
        temp_var_e_var, temp_phis_var = esti_var
        
        self.params.set_phis(temp_phis)
        self.params.set_var_e(temp_var_e)
        self.params.set_phis_var(temp_phis_var)
        self.params.set_var_e_var(temp_var_e_var)
        
        self.ts = ts
        self.get_residuals()
        self.end_ts = np.flip(ts[-self.params.order:])   # 'flip' in order to have the most recent obs. at the beginning of array
        self.end_ts = TransformedTS(self.end_ts)
        
        del(self.prev_x)
        del(self.idx_params)
    
    def predict_at_step(self, prev_x:TransformedTS, steps:np.int64):
        # Note : this function may be cached since it uses a recursion.
        # prev_x must strictly be the size of the order of the model
        # the most recent obs is at the beginning of the array
        # Check if steps > 0.
        current_pred = self.params.phis @ prev_x()
        if steps == 1:
            return current_pred
        else:
            new_prev_x = np.concatenate(([current_pred], prev_x[:-1]))
            new_prev_x = TransformedTS(new_prev_x)
            return self.predict_at_step(new_prev_x, steps - 1)
    
    def rolling_pred(self, prev_x:TransformedTS, steps:np.int64):
        pred = np.empty((steps, ), dtype=np.float64)
        pred = TransformedTS(pred)
        for i in range(steps):
            pred[i] = self.predict_at_step(prev_x, i+1)  # '+1' since it loop start with '0'.
        return pred
    
    @lru_cache(maxsize=64)
    def predict_var_at_step(self, steps:np.int64):
        # Add check if steps > 0.
        current_var_pred = self.params.var_e
        if steps == 1:
            return current_var_pred
        else:
            s = 0.0
            if steps >= self.params.order + 1:
                for i in range(self.params.order):
                    s += (self.params.phis[i]**2.0) * self.predict_var_at_step(steps - i - 1)
            else:
                for i in range(steps-1):
                    s += (self.params.phis[i]**2.0) * self.predict_var_at_step(steps - i - 1)
                
            return s + current_var_pred
        
    def rolling_var_pred(self, steps:np.int64):
        var_pred = np.empty((steps, ), dtype=np.float64)
        for i in range(steps):
            var_pred[i] = self.predict_var_at_step(i)
        return var_pred
        
        
        
        
        
        
        
        
        
        

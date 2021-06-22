import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Models import *

class Trade_Logic:
    model:Model
    sequence:np.ndarray     # correspond to the part of the ts which is visible to the Model (for estim, ...)
    def __init__(self):
        pass
    
    def update_logic(self):
        pass
    
    def open_trade(self):
        pass
    
    def close_trade(self):
        pass

class Trade_random(Trade_Logic):
    counter:np.int8
    def __init__(self, proba_trade:np.float64, duration:np.int8):
        self.proba_trade = proba_trade
        self.duration = duration
        self.is_trade_open = False
        
    def update_logic(self, model:Model, sequence:np.ndarray):
        self.model = model
        self.sequence = sequence
        
        if self.is_trade_open:
            self.counter += 1
        
    def open_trade(self):
        # Return true if wants to open a trade
        # This function will only be called if NO trade is currently open
        if np.random.binomial(1, self.proba_trade) == 1:
            self.counter = 0
            self.is_trade_open = True
            return True
        return False
        
    def close_trade(self):
        # Return true if wants to close a trade
        # This function will only be called if a trade is currently open
        if self.counter >= self.duration:
            self.is_trade_open = False
            return True
        return False
        
class AR_logic(Trade_Logic):
    def __init__(self, threshold_proba:np.float64, n_periods:np.int8, duration:np.int8):
        self.threshold_proba = threshold_proba # Probability of negative retun after 'n_periods' given prediction.
        self.n_periods = n_periods
        self.duration = duration
        self.is_trade_open = False
        
    def update_logic(self, model:Model, sequence:np.ndarray):
        self.model = model
        self.sequence = sequence
        if self.is_trade_open:
            self.counter += 1
        
    def open_trade(self):
        # Note that 'np.cumsum' is only applicable since we are using the log-differenced serie !!
        # Additional care should be taken when using other types of transformations.
        mean_pred = np.sum(self.model.rolling_pred(self.model.end_ts, self.n_periods))
        std_pred = np.sqrt(self.model.rolling_var_pred(self.n_periods))
        std_pred = np.sum(std_pred)
        
        t = self.model.dist.cdf(self.model.end_ts[0], mean_pred, std_pred)
        if self.threshold_proba >=  t:
            self.counter = 0
            self.is_trade_open = True
            return True
        return False
    
    def close_trade(self):
        # Return true if wants to close a trade
        # This function will only be called if a trade is currently open
        if self.counter >= self.duration:
            self.is_trade_open = False
            return True
        return False
        
        
        
        
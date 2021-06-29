import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Models import *
from Transformations import *
from Arrays import *

class Trade_Logic:
    model:Model
    sequence:np.ndarray     # correspond to the part of the ts which is visible to the Model (for estim, ...)
    pipeline:Transform_Pipeline
    def __init__(self, pipe:Transform_Pipeline):
        self.pipe = pipe
    
    def update_logic(self):
        pass
    
    def open_trade(self):
        pass
    
    def close_trade(self):
        pass

class Trade_random(Trade_Logic):
    counter:np.int8
    def __init__(self, pipe:Transform_Pipeline, proba_trade:np.float64, duration:np.int8):
        super().__init__(self, pipe)
        self.proba_trade = proba_trade
        self.duration = duration
        self.is_trade_open = False
        
    def update_logic(self, model:Model, sequence:TransformedTS):
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
    def __init__(self, pipe:Transform_Pipeline, worse_loss:np.float64, n_periods:np.int8, duration:np.int8):
        super().__init__(pipe)
        self.worse_loss = worse_loss # Worse loss (at 95% proba).
        self.n_periods = n_periods
        self.duration = duration
        self.is_trade_open = False
        
    def update_logic(self, model:Model, sequence:TransformedTS):
        self.model = model
        self.sequence = sequence    # Note that 'sequence' is a TransformedTS object;
        if self.is_trade_open:
            self.counter += 1
        
    def open_trade(self):
        mean_pred = self.model.rolling_pred(self.model.end_ts, self.n_periods)  # Mean prediciton in the transformed domain.
        std_pred = self.model.rolling_var_pred(self.n_periods)     # Std prediction in the transformed domain.
        std_pred = std_pred.apply(np.sqrt)
        
        tmp_std = mean_pred() - 1.96 * std_pred()    # Only the lower conf. interval (at 95% gaussian).
        std_pred = TransformedTS(tmp_std)
        
        mean_pred_orig = self.sequence.concatenate(mean_pred)
        mean_pred_orig = self.pipe.reverse_transform(mean_pred_orig)
        
        std_pred_orig = self.sequence.concatenate(std_pred)
        std_pred_orig = self.pipe.reverse_transform(std_pred_orig)
        
        t = (std_pred_orig[-1] - std_pred_orig[-self.n_periods-1])/std_pred_orig[-self.n_periods-1]  # Worse return given prediction.
        if self.worse_loss <  t:
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
        
        
        
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Trade:
    positions:list
    idx_close:np.int64
    PnL:np.float64
    duration:np.int8
    def __init__(self, idx_open:np.int64):
        self.idx_open = idx_open    # when 'idx_open' = 0, it means that the trade 
                                    # is opened at the 'n_period' + 0 time step of the original TS;
        self.positions = []
        self.is_open = True
    
    def add_position(self, idx_open:np.int64, log_return:np.float64):
        new_pos = Position(idx_open, log_return)
        self.positions.append(new_pos)
        
    def close(self):
        self.positions = np.array(self.positions)
        self.idx_close = self.positions[-1].get_idx_open()
        self.is_open = False
        self.__compute_statistics()
        
    def get_is_open(self):
        return self.is_open
        
    def __compute_statistics(self):
        self.__compute_PnL()
        self.__compute_duration()
            
    def __compute_PnL(self):
        self.PnL = 0.0
        for p in self.positions:
            self.PnL += p.get_log_return()
            
    def __compute_duration(self):
        self.duration = self.idx_close - self.idx_open
        
        
    
class Position:
    def __init__(self, idx_open:np.int64, log_return:np.float64):
        self.idx_open = idx_open
        self.log_return = log_return
        
    def get_idx_open(self):
        return self.idx_open
    
    def get_log_return(self):
        return self.log_return
    
    
    
    
    
    
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Statistics import *
from Models import *
from Trade import *
from Trade_logic import *
import Utilities as utils
import multiprocessing
from time import time
from functools import partial
from copy import deepcopy

class WalkForwardBacktester:
    model_history:np.ndarray
    trade_history:list
    PnL:np.ndarray
    def __init__(self, model:Model, trade_logic:Trade_Logic, ts:np.ndarray, n_train:np.int8):
        self.model = model  # Model should already be initialised
        self.trade_logic = trade_logic  # Trade_logic should already be initialised
        self.ts = ts
        self.n_train = n_train
        self.n_periods = len(self.ts) - self.n_train
        self.trade_history = []
        
        if self.n_periods <= 0:
            raise ValueError('Not enough data to perform a walk forward analysis')
        
    def run(self):
        self.model_history = np.empty((self.n_periods,), dtype=Model)
        for i in range(self.n_periods):
            sequence = self.ts[i:i+self.n_train]
            self.__fit_model_to_sequence(sequence, None) if i == 0 else self.__fit_model_to_sequence(sequence, self.model_history[i-1].params.to_array())
            self.model_history[i] = deepcopy(self.model)
            
            self.trade_logic.update_logic(self.model_history[i], sequence)
            
            if self._is_trade_open():
                self.trade_history[-1].add_position(i, sequence[-1]) # 'sequence' must be the log_return of the original TS !
                if self.trade_logic.close_trade():
                    self.trade_history[-1].close()
            else:
                if self.trade_logic.open_trade():
                    self.trade_history.append(Trade(i+1)) # '+1' since the trade is actually opened the next period
                    # Check if last 'Trade' is empty
        self.__compute_PnL()
                    
    def _is_trade_open(self):
        if len(self.trade_history) == 0:
            return False
        else:
            return self.trade_history[-1].get_is_open()
            
    def __fit_model_to_sequence(self, sequence:np.ndarray, init_guess:np.ndarray):
        self.model.fit(sequence,init_guess=init_guess)
    
    def params_history_to_df(self):
        cols = list(self.model_history[0].params.to_dict().keys())
        params_history = np.empty((self.n_periods, len(cols)), dtype=np.float64)
        for i, model in enumerate(self.model_history):
            params_history[i, :] = np.array(list(model.params.to_dict().values()))
            
        return pd.DataFrame(params_history, columns=cols)
    
    def plot_params_history(self):
        params_history = self.params_history_to_df()
        params_history.plot(subplots=True, figsize=(12, 15))

    def __compute_PnL(self):
        self.PnL = np.zeros((self.n_periods,), dtype=np.float64)
        for trade in self.trade_history:
            for position in trade.positions:
                self.PnL[position.idx_open] = position.log_return
    
    def plot_PnL(self):
        fig, ax = plt.subplots(3,1,figsize=(12, 8))
        ax = np.ravel(ax)
        
        ax[0].plot(self.ts[-self.n_periods:])
        ax[1].plot(self.PnL)
        ax[2].plot(np.cumsum(self.PnL))
        
        

def worker2(logic, i):
    init_m = AR(2)
    init_m.set_params(np.array([-0.2, 0.1]), 0.001)
    
    q = []
    m = AR(2)
    for i in range(10):
        data = init_m.sample(3000)
        fwdB = WalkForwardBacktester(m, logic, data, 2750)
        fwdB.run()
        q.append(np.sum(fwdB.PnL))
    return q


if __name__ == '__main__':
    # path_project = 'C:\\Users\\guill\\OneDrive\\Trading\\Python\\Projects'
    # path_data = path_project + '\\Data_Binance\\15m\\ADABTC.csv'
    # data = pd.read_csv(path_data, sep=';', encoding='utf-8', header=0, index_col=0)['Close']
    # data = data.pct_change()
    # data = data.to_numpy()
    # data = np.log1p(data)[1:]
    
    # np.random.seed(123)
    

    
    # logic = Trade_random(0.1, 4)
    logic = AR_logic(0.2, 1, 1)
    
    partial_worker = partial(worker2, logic)
    
    start = time()
    
    # fwdB = WalkForwardBacktester(AR(2), logic, data, 2750)
    # fwdB.run()
    # fwdB.plot_PnL()
    
    # init_m.fit(data)
    # print(init_m.params.to_dict())()
    
    # from Distribution import *
    # d = Normal()
    # d.add_model(init_m)
    
    
    pool = multiprocessing.Pool()
    result = pool.map_async(partial_worker, range(15))
    
    PnL = np.array(result.get()).ravel()
    
    plt.hist(PnL, bins=100)
    
    
    end = time()
    
    print(f'Time elapsed : {end - start}')













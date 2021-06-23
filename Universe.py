import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Models import *
from Backtester import *
from Trade_logic import *
from Dataware import *
import multiprocessing
from functools import partial
from time import time

path_project = 'C:\\Users\\guill\\OneDrive\\Trading\\Python\\Projects'
path_data = path_project + '\\Data_Binance\\15m'
path_stats = path_project + '\\ProbaDingue\\export_all_stats.csv'

class Universe:
    tickers:np.ndarray
    backtests:np.ndarray
    def __init__(self, model:Model, trade_logic:Trade_Logic):
        self.model = model
        self.trade_logic = trade_logic
        self.n_backtest_period = 250
        
        self.filterUniverse()
        
    def getUniverse(self):
        self.tickers = pd.read_csv(path_stats, sep=';', encoding='utf-8', usecols=0)
        
    # Function used to select some cryptos
    def filterUniverse(self):
        # Find a way not to hard code the restrictions (new class ?)
        tmp_data = pd.read_csv(path_stats, sep=';', encoding='utf-8', index_col=0)
        tmp_data = tmp_data[tmp_data.loc[:, 'sample_size_Open'] >= 2000]
        self.tickers = tmp_data.head(10).index
    
    def run_one_backtest(self, ticker:str):
        path_ticker = path_data + '\\' + ticker + '.csv'
        data = pd.read_csv(path_ticker, sep=';', encoding='utf-8', index_col=0)['Close']
        
        # This data transformation will be done in a separate class
        data = data.pct_change()
        data = np.log1p(data)[1:]
        data = data.to_numpy()
        
        fwdB = WalkForwardBacktester(self.model, self.trade_logic, data, len(data) - self.n_backtest_period)
        fwdB.run()
        return (ticker,fwdB)
        
    def run_backtests(self):
        pool = multiprocessing.Pool(4)
        results = pool.map_async(self.run_one_backtest, self.tickers)
        self.backtests = dict(results.get())
        
    def plot_PnLs(self):
        fig, ax = plt.subplots(1,1,figsize=(12, 8))
        for key, value in self.backtests.items():
            ax.plot(np.cumsum(value.PnL))
            
        
if __name__ == '__main__':
    m = AR(1)
    logic = AR_logic(0.2, 1, 1)
    u = Universe(m, logic)
    
    start = time()
    u.run_backtests()
    end = time()
    print(f'Time elapsed : {end - start}')
    
    
    
    
    
    
    
    
    
    
        
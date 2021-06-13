import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Statistics import *

path_project = 'C:\\Users\\guill\\OneDrive\\Trading\\Python\\Projects\\ProbaDingue'
path_data = path_project + '\\binance_data\\15m'

def run_over_files(func, col:str, *args):
    # This function returns a dictionary with the output of the given function for all files.
    list_files = os.listdir(path_data)
    
    d = {}
    for i in range(len(list_files)):
        s = list_files[i].split(".")[0]
        df = pd.read_csv(path_data + '\\' + list_files[i], sep=";", index_col=0)[col]
            
        d[s] = func(df.to_numpy(), *args)
    
    return d

def run_one_file(file:str, func, col:str, *args):
    path_file = path_data + '\\' + file
    df = pd.read_csv(path_file, sep=';', index_col=0)[col]
    return func(df.to_numpy(), *args)

def create_statistics(ts:np.ndarray):
    return Statistics(ts)

def group_statistic_ts(col:str):
    dic = run_over_files(create_statistics, col)
    cols = map(lambda s: s + '_'+col, list(dic.values())[0].cols)
    cols = list(cols)
    d = {}
    for key, value in dic.items():
        d[key] = value.array_from_stats()
    return pd.DataFrame.from_dict(d, orient='index', columns=cols)

def group_all_statistics():
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'N_trades']
    
    df = pd.DataFrame()
    for c in cols:
        df_col = group_statistic_ts(c)
        df = pd.concat([df, df_col], axis=1)
    
    df.index.name = 'Ticker'
    return df

def export_all_statistics():
    path_export = path_project + '\\export_all_stats.csv'
    df = group_all_statistics()
    df.to_csv(path_export, sep=';', float_format='%.15f', encoding='utf-8')
    
def split_all_tickers():
    path_all_stats = path_project + '\\export_all_stats.csv'
    df = pd.read_csv(path_all_stats, sep=';', index_col=0)
    tickers = df.index
    
    d = {}
    for s in tickers:
        d[s] = split_one_ticker(s)
    return pd.DataFrame.from_dict(d, orient='index', columns=['Tick', 'Base'])

def split_one_ticker(ticker:str):
    bases = ['BTC', 'ETH', 'BNB', 'USDT']
    for b in bases:
        splitted = ticker.split(b)
        if len(splitted) == 2:
            if splitted[0] == '':
                return [b, splitted[1]]
            else:
                return [splitted[0], b]
    return ['', '']

def add_split_tickers():
    path_all_stats = path_project + '\\export_all_stats.csv'
    df = pd.read_csv(path_all_stats, sep=';', index_col=0)
    if not 'Base' in df.columns:
        df_tickers = split_all_tickers()
        df = pd.concat([df_tickers, df], axis=1)
        df.to_csv(path_all_stats, sep=';', float_format='%.15f', encoding='utf-8')
    
    
    
    
    
    
    
    
    
    
    
    
    



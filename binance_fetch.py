import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from zipfile import ZipFile

# Add info about file in a csv (which crypto, how many obs, ...)

# Should first 'extract_all_crypto' then 'regroup_csv'

path_project = 'C:\\Users\\guill\\OneDrive\\Trading\\Python\\Projects\\ProbaDingue'
path_klines = path_project + '\\fetch_binance\\data\\spot\\daily\\klines'

def get_crypto_list():
    return os.listdir(path_klines)

def extract_all_crypto(interval:str):
    crypto_list = get_crypto_list()
    
    for crypto in crypto_list:
        print('Extracting ' + crypto)
        extract_one_crypto(crypto, interval)
    
def extract_one_crypto(crypto_name:str, interval:str):
    path_crypto = path_klines + '\\' + crypto_name 
    path_interval = path_crypto + '\\' + interval
    zipFiles_list = os.listdir(path_interval)
    
    if len(zipFiles_list) == 0:
        shutil.rmtree(path_crypto)
        return
    
    path_extracted_csv = path_crypto + '\\' + interval + '_csv'
    
    # If the folder already exists, we delete it with all its files !
    if os.path.exists(path_extracted_csv):
        shutil.rmtree(path_extracted_csv)
    os.mkdir(path_extracted_csv)
    
    for file in zipFiles_list:
        path_zip = path_interval + '\\' + file
        extract_zip(path_zip, path_extracted_csv)
    
def extract_zip(path_zip, extract_to):
    # Extract zip file to the working directory !
    
    with ZipFile(path_zip, 'r') as z:
        z.extractall(path=extract_to)

def regroup_csv(interval:str):
    crypto_list = get_crypto_list()
    
    path_group = path_project + '\\binance_data\\' + interval
    if os.path.exists(path_group):
        shutil.rmtree(path_group)
    os.mkdir(path_group)
    
    for crypto in crypto_list:
        print('Grouping ' + crypto)
        regroup_csv_one_crypto(crypto, interval)
        
# Should sort final dataframe with 'Open_time' column
    
def regroup_csv_one_crypto(crypto_name:str, interval:str):
    path_crypto = path_klines + '\\' + crypto_name
    path_interval = path_crypto + '\\' + interval + '_csv'
    csv_list = os.listdir(path_interval)
    
    head = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'N_trades',
        'Taker_buy_base_vol', 'Taker_buy_quote_vol', 'Ignore']
    
    df_list = []
    
    for csv in csv_list:
        path_csv = path_interval + '\\' + csv
    
        df_list.append(pd.read_csv(path_csv, header=0, names=head))
        
    df = pd.concat(df_list, ignore_index=True)
    
    path_regrouped = path_project + '\\binance_data\\' + interval + '\\' + crypto_name + '.csv'
    
    df.to_csv(path_regrouped, sep=';', float_format='%.15f', encoding='utf-8')
    
    
    
    
    
    
    
    
    
    
    
    
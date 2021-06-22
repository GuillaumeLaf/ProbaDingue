import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from zipfile import ZipFile

# Add info about file in a csv (which crypto, how many obs, ...)

# Should first 'extract_all_crypto' then 'regroup_csv' (after the crypto has been downloaded).

path_project = 'C:\\Users\\guill\\OneDrive\\Trading\\Python\\Projects'
path_klines = path_project + '\\Data_Binance_Get\\data\\spot\\daily\\klines'

def get_crypto_list():
    return os.listdir(path_klines)

def extract_all_crypto(interval:str):
    crypto_list = get_crypto_list()
    
    for crypto in crypto_list:
        print('Extracting ' + crypto)
        extract_one_crypto(crypto, interval)
    
def extract_one_crypto(crypto_name:str, interval:str):
    # Extract the csv file from the zip file for a given crypto
    path_crypto = path_klines + '\\' + crypto_name 
    path_interval = path_crypto + '\\' + interval
    zipFiles_list = os.listdir(path_interval)
    
    # If file is empty, we delete it (may happen when there is no file to download from Binance)
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
    # Extract zip file to the given file.
    with ZipFile(path_zip, 'r') as z:
        z.extractall(path=extract_to)

def regroup_csv(interval:str):
    # Once the csv has been extracted from the zip files, 
    # We regroup each separate csv file for a given crypto in a unique one. This function does it for all crypto.
    crypto_list = get_crypto_list()
    
    path_group = path_project + '\\Data_Binance\\' + interval
    if os.path.exists(path_group):
        shutil.rmtree(path_group)
    os.mkdir(path_group)
    
    for crypto in crypto_list:
        print('Grouping ' + crypto)
        regroup_csv_one_crypto(crypto, interval)
        
# Should sort final dataframe with 'Open_time' column
    
def regroup_csv_one_crypto(crypto_name:str, interval:str):
    # Regroup the separate csv file for a given crypto in a unique csv.
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
    
    path_regrouped = path_project + '\\Data_Binance\\' + interval + '\\' + crypto_name + '.csv'
    
    df.to_csv(path_regrouped, sep=';', float_format='%.15f', encoding='utf-8')
    
    
    
    
    
    
    
    
    
    
    
    
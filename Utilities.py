import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Models as models

def plot_corrFunc(corr_lags:np.ndarray):
    """
    Function used as a pattern to plot the Autocorrelation and Autocovariance function

    Parameters
    ----------
    corr_lags : np.ndarray
        DESCRIPTION.

    Returns
    -------
    None.

    """
    max_lag = len(corr_lags)
    fig, ax = plt.subplots(1,1,figsize=(12, 8))
    ax.plot(np.arange(1,max_lag+1), corr_lags, 'o')
    ax.axhline(0.0, color='black', alpha=0.2)
    
    x_ticks = np.arange(0,max_lag,5)
    x_ticks[0] += 1
    ax.set_xticks(x_ticks)
    
    for i in range(max_lag):
        ax.vlines(x=i+1, ymin=0.0, ymax=corr_lags[i], color='black')

def acf(ts:np.ndarray, max_lag:np.int8):
    """
    Compute the Autocorrelation function of a time series 'ts' up to lag 'max_lag'

    Parameters
    ----------
    ts : np.ndarray
        DESCRIPTION.
    max_lag : np.int8
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # Restrict 'max_lag' function length of ts
    out = np.empty((max_lag,), dtype=np.float64)
    for i in range(1, max_lag+1):
        out[i-1] = (ts[:-i] @ ts[i:]) / len(ts)
    
    return out / ((ts @ ts) / len(ts))

def plot_acf(ts:np.ndarray, max_lag:np.int8=20):
    plot_corrFunc(acf(ts, max_lag))
        
def pacf(ts:np.ndarray, max_lag:np.int8):
    """
    Compute the Partial Autocorrelation function of a time series 'ts' up to lag 'max_lag'

    Parameters
    ----------
    ts : np.ndarray
        DESCRIPTION.
    max_lag : np.int8
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # Restrict 'max_lag' function length of ts
    out = np.empty((max_lag,), dtype=np.float64)
    
    out[0] = (ts[:-1] @ ts[1:]) / len(ts)
    
    # x_i will be (n x T-max_lag)
    x_i = ts[max_lag-1:-1].reshape(1, -1)

    for i in range(2, max_lag+1):
        # beta_0 is (n x 1)
        beta_0 = np.linalg.inv(x_i @ x_i.T) @ (x_i @ ts[max_lag:])
        resid_0 = ts[max_lag:] - (x_i.T @ beta_0)
        
        beta_i = np.linalg.inv(x_i @ x_i.T) @ (x_i @ ts[max_lag-i:-i])
        resid_i = ts[max_lag-i:-i] - (x_i.T @ beta_i)
        
        out[i-1] = (resid_0 @ resid_i) / len(resid_0)
        
        x_i = np.concatenate((x_i, ts[max_lag-i:-i].reshape(1, -1)), axis=0)
        
    return out / ((ts @ ts) / len(ts))

def plot_pacf(ts:np.ndarray, max_lag:np.int8=20):
    plot_corrFunc(pacf(ts, max_lag))


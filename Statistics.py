import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Statistics:
    mean:np.float64
    std:np.float64
    percentiles:np.ndarray
    sample_size:np.int64
    cols:list
    def __init__(self, ts:np.ndarray):
        self.ts = ts
        self.cols = []
        self.__initialize()
        
    def array_from_stats(self):
        single_stats = np.array([self.sample_size, self.mean, self.std])
        multiple_stats = np.concatenate((single_stats, self.percentiles))
        return multiple_stats
        
    def __initialize(self):
        self.__get_sample_size()
        self.__get_avg()
        self.__get_std()
        self.__get_percentiles()
        
    def __get_avg(self):
        self.cols.append('mean')
        self.mean = np.mean(self.ts)
        
    def __get_std(self):
        self.cols.append('std')
        self.std = np.std(self.ts)
        
    def __get_percentiles(self):
        percentile = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        self.percentiles = np.empty((len(percentile),), dtype=np.float64)
        for i in range(len(percentile)):
            self.cols.append('p'+str(percentile[i]))
            self.percentiles[i] = np.percentile(self.ts, percentile[i])
            
    def __get_sample_size(self):
        self.cols.append('sample_size')
        self.sample_size = len(self.ts)
        


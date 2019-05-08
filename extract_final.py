""" This code extracts temperature, humidity, and pressure information from 5 west coast cities and cleans the
data by discarding outlier points that are either due to sensor malfunction or data collection error. It then
saves the cleaned data in csv format for further processing.

Original dataset:
    https://www.kaggle.com/selfishgene/historical-hourly-weather-data
"""
import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt

# Read in temperature, humidity, and pressure
temperature = pd.read_csv('temperature.csv', usecols=["datetime","Vancouver", "Portland", "San Francisco", "Seattle", "Los Angeles"])
humidity = pd.read_csv('humidity.csv', usecols=["Vancouver", "Portland", "San Francisco", "Seattle", "Los Angeles"])
pressure = pd.read_csv('pressure.csv', usecols=["Vancouver", "Portland", "San Francisco", "Seattle", "Los Angeles"])

# Create a single dataframe from the information we are interested in.
df = pd.concat([temperature,humidity,pressure], axis=1)
df.columns = ['datetime', 't-Van', 't-Port', 't-SF', 't-Sea', 't-LA', 'h-Van', 'h-Port', 'h-SF', 'h-Sea', 'h-LA', 'p-Van', 'p-Port', 'p-SF', 'p-Sea', 'p-LA']

kw = dict(marker='o', linestyle='none', color='r', alpha=0.3)

# Fix the pressures; for readings that deviate too far from the local median, discard the datapoint
# because it is likely a sensor error (or a data scraping error in the dataset)
fix_cols = ['p-Van', 'p-Port', 'p-SF', 'p-Sea', 'p-LA']
for c in fix_cols:
    threshold = 70  # If the pressure differs by more than 70 millibars from the 5000-hour window, discard the point
    df['pandas'] = Series.rolling(df[c], window=5000, min_periods=10, center=True).median()
    
    difference = np.abs(df[c] - df['pandas'])
    outlier_idx = difference > threshold
    
    df.loc[outlier_idx,c] = np.nan
    
    # If the pressure differs by more thna 40 millibars in a 400-hour window, discard the point
    threshold = 40
    df['pandas'] = Series.rolling(df[c], window=400, min_periods=5, center=True).median()
    
    difference = np.abs(df[c] - df['pandas'])
    outlier_idx = difference > threshold
    
    df.loc[outlier_idx,c] = np.nan
    
    num_outliers = 1
    threshold = 10
    while num_outliers > 0:
        print(num_outliers)
        # Using a window of 5 hours, discard outliers of more than 10 millibars from the median
        # of those 5 points (minimum of 3 points in window to take action)
        df['pandas'] = Series.rolling(df[c], window=5, min_periods=3, center=True).median()
        
        difference = np.abs(df[c] - df['pandas'])
        outlier_idx = difference > threshold
        
        df.loc[outlier_idx,c] = np.nan
        num_outliers = np.sum(outlier_idx.get_values())
    
    # Interpolate missing values with a linear interpolation
    df = df.interpolate(method='linear', axis=0, limit_direction='both')
    bad_idx = df.index[df[c].diff().abs().ge(20)]

# Fix outliers in temperatures
fix_cols = ['t-Van', 't-Port', 't-SF', 't-Sea', 't-LA']
for c in fix_cols:
    
    threshold = 10  # Discard points that are more than 10 C away from the two neighboring hours.
    df['pandas'] = Series.rolling(df[c], window=3, center=True).median()
    
    difference = np.abs(df[c] - df['pandas'])
    outlier_idx = difference > threshold
    
    df.loc[outlier_idx,c] = np.nan
    num_outliers = np.sum(outlier_idx.get_values())
    
    # Interpolate missing values.
    df = df.interpolate(method='linear', axis=0, limit_direction='both')
    bad_idx = df.index[df[c].diff().abs().ge(20)]

## Replace remaining with the following searchable number, -99999 so we can discard sections of
## data at the beginning and end of the time period that cannot be interpolated.
df = df.fillna(-99999)
print(df.head(3))
df.to_csv('final_dataset/west_coast_weather.csv', index=False)

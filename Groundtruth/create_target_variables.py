import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def costomize_rolling(df, var):
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=14)
    # compute 'forward' mean
    df[var + '_rolling'] = df[var].rolling(window=indexer, min_periods=14).mean()
    # shift to make week 3 & 4 ground truth
    df[var + '_shift'] = df[var + '_rolling'].shift(-14)
    return df


# read raw data
df_tmp2m = pd.read_hdf('tmp2m_western_us_updated.h5')
df_tmp2m = df_tmp2m.reset_index()
df_tmp2m['month'] = df_tmp2m.start_date.dt.month
df_tmp2m['day'] = df_tmp2m.start_date.dt.day
# read climatology
climo = pd.read_hdf('climo_all.h5')
climo = climo.drop(['tmp2m_mean_raw', 'tmp2m_std_raw'], axis=1)
# compute anomalies
data = df_tmp2m.merge(climo, on=['lat', 'lon', 'month', 'day'], how='left')
data['anom'] = data['tmp2m'] - data['tmp2m_mean_smooth']
data = data.set_index(['lat', 'lon', 'start_date'])
df_tmp2m_all_rolling = data.groupby(['lat', 'lon']).apply(lambda df: costomize_rolling(df, 'anom'))
df_tmp2m_all_rolling = df_tmp2m_all_rolling.dropna()
truth = df_tmp2m_all_rolling['anom_shift'].to_frame()
truth.columns = ['target']
truth.to_hdf('tmp2m_western_us_anom_rmm.h5', key='data')

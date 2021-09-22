import numpy as np
import pandas as pd
import xarray as xr
import scipy.io


def get_std(df):
    x = df['tmp2m'] - df['tmp2m_mean_smooth']
    x = x.values
    std = np.sqrt(np.mean(x**2))
    df['tmp2m_std_smooth'] = std
    return df


climo_smooth = scipy.io.loadmat('climo_smooth.mat')['climo']
climo_smooth = climo_smooth[:, :, 31:-31]
climo_new = xr.open_dataset("climo_2d.nc")
climo_new['tmp2m_mean'].values = climo_smooth
df_climo = climo_new.to_dataframe()
df_climo = df_climo.dropna()
df_climo = df_climo.reset_index()
df_climo['month'] = df_climo.start_date.dt.month
df_climo['day'] = df_climo.start_date.dt.day
df_climo = df_climo[['lat', 'lon', 'month', 'day', 'tmp2m_mean']]
tmp2m_train = pd.read_hdf('tmp2m_train_temp.h5')
tmp2m_train_new = tmp2m_train.rename(columns={'tmp2m_mean': 'tmp2m_mean_raw', 'tmp2m_std': 'tmp2m_std_raw'})
tmp2m_train_new = tmp2m_train_new.merge(df_climo, on=['lat', 'lon', 'month', 'day'], how='left')
tmp2m_train_new = tmp2m_train_new.rename(columns={'tmp2m_mean': 'tmp2m_mean_smooth'})
tmp2m_train_new = tmp2m_train_new.groupby(['lat', 'lon', 'month', 'day']).apply(lambda df: get_std(df))
climo_all = tmp2m_train_new[tmp2m_train_new.start_date.dt.year == 2000]
column_names = ['lat', 'lon', 'month', 'day', 'tmp2m_mean_raw', 'tmp2m_std_raw', 'tmp2m_mean_smooth', 'tmp2m_std_smooth']
climo_all = climo_all[column_names]
climo_all = climo_all.set_index(['lat', 'lon', 'month', 'day']).sort_index()
climo_all = climo_all.reset_index()
climo_all.to_hdf('climo_all.h5', key='data')

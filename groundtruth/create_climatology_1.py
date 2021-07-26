import numpy as np
import pandas as pd
import xarray as xr
import scipy.io


tmp2m = pd.read_hdf('tmp2m_western_us_updated.h5').reset_index()
tmp2m_train = tmp2m[tmp2m.start_date >= '1990-01-01']
tmp2m_train = tmp2m_train[tmp2m_train.start_date < '2017-07-01']
tmp2m_train = tmp2m_train.reset_index(drop=True)

tmp2m_train['month'] = tmp2m_train['start_date'].dt.month
tmp2m_train['day'] = tmp2m_train['start_date'].dt.day

tmp2m_train['tmp2m_mean'] = tmp2m_train.groupby(['lat', 'lon', 'month', 'day'])['tmp2m'].transform('mean')
tmp2m_train['tmp2m_std'] = tmp2m_train.groupby(['lat', 'lon', 'month', 'day'])['tmp2m'].transform('std')
tmp2m_train.to_hdf('tmp2m_train_temp.h5', key='data')

climo = tmp2m_train[tmp2m_train.start_date.dt.year == 2000]
climo = climo.reset_index(drop=True)
climo = climo[['lat', 'lon', 'start_date', 'month', 'day', 'tmp2m_mean', 'tmp2m_std']]

climo_s = climo.set_index(['lat', 'lon', 'start_date'])
climo_s = climo_s['tmp2m_mean']

# add dec and jan at the beginning and end for smoothing
climo_2d = xr.DataArray.from_series(climo_s)
climo_2d.to_netcdf("climo_2d.nc")

dec = climo_2d.values[:, :, -31:]
jan = climo_2d.values[:, :, :31]
acm_climo = np.concatenate((dec, climo_2d.values, jan), axis=2)
scipy.io.savemat('acm_climo.mat', dict(climo_raw=acm_climo))

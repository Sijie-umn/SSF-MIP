from netCDF4 import Dataset
import pandas as pd
import numpy as np

for model in ['NCEP', 'GMAO']:

    tmp2m_week34 = pd.read_hdf('tmp2m_{}.h5'.format(model)).reset_index()
    tmp2m_week34_climo = pd.read_hdf('tmp2m_{}_climo.h5'.format(model))
    tmp2m_week34_climo['climo'] = tmp2m_week34_climo['tmp2m']
    tmp2m_week34_climo = tmp2m_week34_climo.drop('tmp2m', axis=1)
    tmp2m_week34['month'] = tmp2m_week34.start_date.dt.month
    tmp2m_week34['day'] = tmp2m_week34.start_date.dt.day
    tmp2m_week34 = tmp2m_week34.merge(tmp2m_week34_climo, on=['lat', 'lon', 'month', 'day'], how='left')
    tmp2m_week34['{}_anom'.format(model)] = tmp2m_week34['tmp2m'] - tmp2m_week34['climo']
    tmp2m_week34 = tmp2m_week34.drop(['month', 'day'], axis=1)

    tmp2m_week34 = tmp2m_week34.rename(columns={"tmp2m": "tmp2m_week34", "climo": "climo_week34", "{}_anom".format(model): "{}_anom_week34".format(model)})
    tmp2m_week34 = tmp2m_week34.set_index(['lat', 'lon', 'start_date'])
    tmp2m_week34 = tmp2m_week34.sort_index()
    tmp2m_week34.to_hdf('tmp2m_{}_week34_forecast.h5'.format(model), key='data')

    tmp2m_week34 = pd.read_hdf('tmp2m_{}_hindcast.h5'.format(model)).reset_index()
    tmp2m_week34_climo = pd.read_hdf('tmp2m_{}_climo.h5'.format(model))
    tmp2m_week34_climo['climo'] = tmp2m_week34_climo['tmp2m']
    tmp2m_week34_climo = tmp2m_week34_climo.drop('tmp2m', axis=1)
    tmp2m_week34['month'] = tmp2m_week34.start_date.dt.month
    tmp2m_week34['day'] = tmp2m_week34.start_date.dt.day
    tmp2m_week34 = tmp2m_week34.merge(tmp2m_week34_climo, on=['lat', 'lon', 'month', 'day'], how='left')
    tmp2m_week34['{}_anom'.format(model)] = tmp2m_week34['tmp2m'] - tmp2m_week34['climo']
    tmp2m_week34 = tmp2m_week34.drop(['month', 'day'], axis=1)

    tmp2m_week34 = tmp2m_week34.rename(columns={"tmp2m": "tmp2m_week34", "climo": "climo_week34", "{}_anom".format(model): "{}_anom_week34".format(model)})
    tmp2m_week34 = tmp2m_week34.set_index(['lat', 'lon', 'start_date'])
    tmp2m_week34 = tmp2m_week34.sort_index()
    tmp2m_week34.to_hdf('tmp2m_{}_week34_hindcast.h5'.format(model), key='data')

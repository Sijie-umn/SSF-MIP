import pandas as pd
import numpy as np
import scipy.io
from netCDF4 import Dataset

# create tmp2m for each year
for year in range(1979, 2021):
    print('preprocessing year:', year)
    lat = np.arange(-90, 91)
    lon = np.arange(0, 360)
    times = pd.date_range('{}-01-01'.format(year), '{}-12-31'.format(year))
    t_regrid = scipy.io.loadmat('regrid_{}.mat'.format(year))['regrid']
    m = np.swapaxes(t_regrid, 0, 1)
    lat, lon, dates = np.meshgrid(lat, lon, times, indexing='ij')
    dates = dates.flatten()
    lat = lat.flatten()
    lon = lon.flatten()
    arrays = [lat, lon, dates]
    tuples = list(zip(*arrays))
    indexnames = ['lat', 'lon', 'start_date']
    index = pd.MultiIndex.from_tuples(tuples, names=indexnames)
    s = pd.Series(m.flatten(), index=index)
    s = s.to_frame()
    s.columns = ['tmp2m']
    s = s.dropna()
    s.to_hdf('tmp2m.{}.verif.h5'.format(year), key='data')

# create tmp2m covering western us
# combine all the western us data and save it into one pandas dataframe

w_us = pd.read_hdf('western_us_mask.h5')  # a pandas DataFrame with lat and lon covering western us
df_tmp2m = pd.DataFrame()
for year in range(1979, 2021):
    print('year: ', year)
    data = pd.read_hdf('tmp2m.{}.verif.h5'.format(year))
    data = data.reset_index()
    data['lat'] = data['lat'].astype('float')
    data['lon'] = data['lon'].astype('float')
    data = w_us.merge(data, on=['lat', 'lon'], how='inner')
    # data.to_hdf('tmp2m.{}.western_us.h5'.format(year), key='data')
    df_tmp2m = df_tmp2m.append(data)
df_tmp2m.to_hdf('tmp2m_western_us.h5', key='data')

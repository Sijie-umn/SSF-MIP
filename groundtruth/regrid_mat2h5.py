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

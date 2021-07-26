from netCDF4 import Dataset
import pandas as pd
import numpy as np

model = 'GMAO'  # or 'GMAO'

if model == 'NCEP':
    path = '/glade/work/bcash/ssf/subx/forecast/tas2m/daily/full/NCEP-CFSv2/'
    NCEP_date = []
    for date in pd.date_range('2017-07-01', '2019-12-31'):
        data = Dataset(path + 'tas_2m_NCEP-CFSv2_{}.emean.daily.nc'.format(date.strftime('%Y%m%d')))
        num = np.count_nonzero(np.isnan(data.variables['tas'][0]))
        if num != 65160:
            # print(date, num)
            NCEP_date.append(date)
    NCEP_date = pd.Series(NCEP_date)
    NCEP_date.to_hdf('NCEP_date.h5', key='date')
elif model == 'GMAO':
    path = '/glade/work/bcash/ssf/subx/forecast/tas2m/daily/full/GMAO-GEOS_V2p1/'
    GMAO_date = []
    for date in pd.date_range('2017-07-25', '2020-09-28'):
        data = Dataset(path + 'tas_2m_GMAO-GEOS_V2p1_{}.emean.daily.nc'.format(date.strftime('%Y%m%d')))
        num = np.count_nonzero(np.isnan(data.variables['tas'][0]))
        if num != 65160:
            # print(date, num)
            GMAO_date.append(date)
    GMAO_date = pd.Series(GMAO_date)
    GMAO_date.to_hdf('GMAO_date.h5', key='date')

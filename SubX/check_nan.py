from netCDF4 import Dataset
import pandas as pd
import numpy as np

ncep_path = '/SubX/forecast/tas2m/daily/full/NCEP-CFSv2/'  # the path where the raw data from NCEP-CFSv2 is saved
gmao_path = '/SubX/forecast/tas2m/daily/full/GMAO-GEOS_V2p1/'

for model in ['NCEP', 'GMAO']:
    if model == 'NCEP':
        path = ncep_path
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
        path = gmao_path
        GMAO_date = []
        for date in pd.date_range('2017-07-25', '2020-09-28'):
            data = Dataset(path + 'tas_2m_GMAO-GEOS_V2p1_{}.emean.daily.nc'.format(date.strftime('%Y%m%d')))
            num = np.count_nonzero(np.isnan(data.variables['tas'][0]))
            if num != 65160:
                # print(date, num)
                GMAO_date.append(date)
    GMAO_date = pd.Series(GMAO_date)
    GMAO_date.to_hdf('GMAO_date.h5', key='date')

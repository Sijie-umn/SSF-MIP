from netCDF4 import Dataset
import pandas as pd
import numpy as np

# model = 'GMAO'  # 'NCEP'

western_us = pd.read_hdf('western_us_mask.h5')

for model in ['GMAO', 'NCEP']:
    date = pd.date_range('2020-01-01', '2020-12-31')  # use 2020 as an example for 365 days
    if model == 'GMAO':
        path = 'hindcast/tas2m/daily/climo/GMAO-GEOS_V2p1/'  # the path where the GMAO hindcasts are saved
        climo_file = 'tmp2m_GMAO_climo.h5'
    elif model == 'NCEP':
        path = 'hindcast/tas2m/daily/climo/NCEP-CFSv2/'  # the path where the NCEP hindcasts are saved
        climo_file = 'tmp2m_NCEP_climo.h5'

    for t in date:
        year = t.year
        month = t.month
        day = t.day
        if model == 'GMAO':
            file_name = 'tas_GMAO-GEOS_V2p1_{}.climo.p.nc'.format(t.strftime('%m%d'))
        elif model == 'NCEP':
            file_name = 'tas_NCEP-CFSv2_{}.climo.p.nc'.format(t.strftime('%m%d'))
        d1 = Dataset(path + file_name)
        if t == pd.Timestamp('2020-01-01'):
            lon_o = d1.variables['lon'][:]
            # lon_o[180:] = lon_o[180:] - 360
            lat, lon = np.meshgrid(d1.variables['lat'], lon_o, indexing='ij')
            lat_f = lat.flatten()
            lon_f = lon.flatten()
            lat_f = lat_f.reshape(lat_f.shape[0], 1)
            lon_f = lon_f.reshape(lon_f.shape[0], 1)
            tas = np.mean(d1.variables['tas'][14:28, :, :], axis=0)
            tas = tas.flatten().reshape(lon_f.shape[0], 1)
            data_one_day = np.concatenate((lat_f, lon_f, tas), axis=1)
            data_one_day = data_one_day[~np.isnan(data_one_day).any(axis=1)]
            data_temp = pd.DataFrame(data_one_day, columns=['lat', 'lon', 'tmp2m'])
            data_temp = data_temp.merge(western_us, on=['lat', 'lon'], how='inner')
            data_temp['month'] = month
            data_temp['day'] = day
            data = data_temp
        else:
            tas = np.mean(d1.variables['tas'][14:28, :, :], axis=0)
            tas = tas.flatten().reshape(lon_f.shape[0], 1)
            data_one_day = np.concatenate((lat_f, lon_f, tas), axis=1)
            data_one_day = data_one_day[~np.isnan(data_one_day).any(axis=1)]
            data_temp = pd.DataFrame(data_one_day, columns=['lat', 'lon', 'tmp2m'])
            data_temp = data_temp.merge(western_us, on=['lat', 'lon'], how='inner')
            data_temp['month'] = month
            data_temp['day'] = day
            data = data.append(data_temp)
    data.to_hdf(climo_file, key='data')

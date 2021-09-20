from netCDF4 import Dataset
import pandas as pd
import numpy as np

western_us = pd.read_hdf('western_us.h5')
model = 'GMAO'  # 'NCEP'

if model == 'NCEP':
    date = pd.date_range('2017-07-01', '2017-09-30')  # can be changed to date file
    date = date.append(pd.date_range('2018-01-16', '2019-03-15'))
    path = '/glade/work/bcash/ssf/subx/forecast/tas2m/daily/full/NCEP-CFSv2/'
    model_name = 'NCEP-CFSv2'
    first_date = pd.Timestamp('2017-07-01')

elif model == 'GMAO':
    date = pd.date_range('2017-07-25', '2020-09-28')
    path = '/glade/work/bcash/ssf/subx/forecast/tas2m/daily/full/GMAO-GEOS_V2p1/'
    model_name = 'GMAO-GEOS_V2p1'
    first_date = pd.Timestamp('2017-07-25')

# date = pd.date_range('2017-07-25', '2020-09-28')
# path = '/glade/work/bcash/ssf/subx/forecast/tas2m/daily/full/GMAO-GEOS_V2p1/'
for e in range(1, 5):
    for t in date:
        year = t.year
        month = t.month
        day = t.day
        file_name = 'tas_2m_{}_{}{:02d}{:02d}.e{}.daily.nc'.format(model_name, year, month, day, e)
        d1 = Dataset(path + file_name)
        if t == first_date:
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
            data_temp['start_date'] = pd.Timestamp('{}{:02d}{:02d}'.format(year, month, day))
            data = data_temp
        else:
            tas = np.mean(d1.variables['tas'][14:28, :, :], axis=0)
            tas = tas.flatten().reshape(lon_f.shape[0], 1)
            data_one_day = np.concatenate((lat_f, lon_f, tas), axis=1)
            data_one_day = data_one_day[~np.isnan(data_one_day).any(axis=1)]
            data_temp = pd.DataFrame(data_one_day, columns=['lat', 'lon', 'tmp2m'])
            data_temp = data_temp.merge(western_us, on=['lat', 'lon'], how='inner')
            data_temp['start_date'] = pd.Timestamp('{}{:02d}{:02d}'.format(year, month, day))
            data = data.append(data_temp)
    data = data.set_index(['lat', 'lon', 'start_date'])
    data.sort_index(inplace=True)
    data.to_hdf('tmp2m_{}_e{}.h5'.format(model, e), key='data')

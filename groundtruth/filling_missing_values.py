import pandas as pd
from datetime import datetime, timedelta

# make sure only keep the tmp2m over western us (508 grid points)
data = pd.read_hdf('tmp2m_western_us.h5')
western_us = data[data.start_date == '2019-01-01'][['lat', 'lon']]
data_updated = western_us.merge(data, on=['lat', 'lon'], how='inner')
data_test = data_updated.set_index(['lat', 'lon', 'start_date'])

# the list of years with missing values for tmp2m
years = [1981, 1983, 1984, 1985, 1986, 1992]

idx = pd.IndexSlice
data = data_test.sort_index()
for year in years:
    print(year)
    date_missing = []
    date_partial_missing = []
    submission_dates = pd.date_range('{}-01-01'.format(year), '{}-12-31'.format(year))
    submission_dates = ['{}-{:02d}-{:02d}'.format(date.year, date.month, date.day) for date in submission_dates]
    for date in submission_dates:
        one_day = data.loc[idx[:, :, date], :]
        if one_day.shape[0] != 508:
            # print(date, one_day.shape[0])
            if one_day.shape[0] == 0:
                date_missing.append(date)
            else:
                date_partial_missing.append(date)

    print('checking is done')
    for date in date_missing[::-1]:
        # date_past=pd.Timestamp(date)-pd.Timedelta(days=1)
        date_next = pd.Timestamp(date) + pd.Timedelta(days=1)
        # one_day_past=data.loc[idx[:,:,date_past],:]
        one_day_next = data.loc[idx[:, :, date_next], :]
        # one_day_past=one_day_past.reset_index()
        one_day_next = one_day_next.reset_index()
        # one_day_next['tmp2m']=(one_day_past['tmp2m']+one_day_next['tmp2m'])/2
        one_day_next['start_date'] = pd.to_datetime(date)
        one_day = one_day_next.set_index(['lat', 'lon', 'start_date'])
        data = pd.concat((data, one_day))
        data = data.sort_index()

data.to_hdf('tmp2m_western_us_updated.h5', key='data')

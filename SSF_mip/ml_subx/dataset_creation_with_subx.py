import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import importlib
import pickle


def save_results(filename, results):
    """save a pickle file

    Args:
    filename: the path + file name for the file to be saved
    results: the data array to be saved
    """
    with open(filename, 'wb') as fh:
        pickle.dump(results, fh)


subx_path = '../../subx/'  # path where subx data are saved
data_path = '../data/'
for model in ['GMAO', 'NCEP']:
    # create dataset
    cov = pd.read_hdf(data_path + 'covariates_all_pc10.h5')
    subx_hindcast = pd.read_hdf(subx_path + 'tmp2m_{}_week34_hindcast.h5'.format(model))
    subx_forecast = pd.read_hdf(subx_path + 'tmp2m_{}_week34_forecast.h5'.format(model))
    subx = subx_hindcast.append(subx_forecast).sort_index()
    subx = subx['{}_anom_week34'.format(model)].to_frame()
    subx_dates = subx.reset_index().start_date.drop_duplicates()
    truth = pd.read_hdf(data_path + 'tmp2m_western_us_anom_rmm.h5').sort_index()
    idx = pd.IndexSlice

    # create train validation index
    train_val_index = {}
    train_range = 12
    gap = 28
    for val_year in [2011, 2012, 2013, 2014, 2015]:
        for val_month in range(1, 13):
            test_time_index = subx_dates[(subx_dates.dt.month == val_month) & (subx_dates.dt.year == val_year)].reset_index(drop=True)
            test_start = test_time_index[0]
            train_end = test_start - pd.DateOffset(days=gap)
            train_start = train_end - pd.DateOffset(years=train_range)
            train_time_index = subx_dates[(subx_dates >= train_start) & (subx_dates <= train_end)].reset_index(drop=True)
            train_val_index['{}-{:02d}'.format(val_year, val_month)] = {'train': train_time_index, 'test': test_time_index}

    # create train test index
    train_test_index = {}
    train_range = 18
    gap = 28
    for val_year in [2017, 2018, 2019, 2020]:
        if model == 'GMAO':
            if val_year == 2017:
                month_range = range(7, 13)
            elif val_year == 2020:
                month_range = range(1, 7)
            else:
                month_range = range(1, 13)
        elif model == 'NCEP':
            if val_year == 2017:
                month_range = range(7, 10)
            elif val_year == 2019:
                month_range = range(1, 4)
            elif val_year == 2018:
                month_range = range(1, 13)
            else:
                month_range = []
        for val_month in month_range:
            test_time_index = subx_dates[(subx_dates.dt.month == val_month) & (subx_dates.dt.year == val_year)].reset_index(drop=True)
            test_start = test_time_index[0]
            train_end = test_start - pd.DateOffset(days=gap)
            train_start = train_end - pd.DateOffset(years=train_range)
            train_time_index = subx_dates[(subx_dates >= train_start) & (subx_dates <= train_end)].reset_index(drop=True)
            train_test_index['{}-{:02d}'.format(val_year, val_month)] = {'train': train_time_index, 'test': test_time_index}

    # create train validation sets
    # create train validation
    for val_year in [2011, 2012, 2013, 2014, 2015]:
        for val_month in range(1, 13):
            train_index = train_val_index['{}-{:02d}'.format(val_year, val_month)]['train']
            test_index = train_val_index['{}-{:02d}'.format(val_year, val_month)]['test']
            train_X = [cov.loc[train_index].values, subx.loc[idx[:, :, train_index], :].unstack(level=[0, 1]).values]
            test_X = [cov.loc[test_index].values, subx.loc[idx[:, :, test_index], :].unstack(level=[0, 1]).values]
            train_y = truth.loc[idx[:, :, train_index], :].unstack(level=[0, 1]).values
            test_y = truth.loc[idx[:, :, test_index], :].unstack(level=[0, 1]).values
            save_results('random_cv/train_y_subx_{}_forecast{}_{}.pkl'.format(val_year, val_month, model), train_y)
            save_results('random_cv/val_y_subx_{}_forecast{}_{}.pkl'.format(val_year, val_month, model), test_y)
            save_results('random_cv/train_X_subx_{}_forecast{}_{}.pkl'.format(val_year, val_month, model), train_X)
            save_results('random_cv/val_X_subx_{}_forecast{}_{}.pkl'.format(val_year, val_month, model), test_X)

    # create train test sets
    # create train test
    for val_year in [2017, 2018, 2019, 2020]:
        if model == 'GMAO':
            if val_year == 2017:
                month_range = range(7, 13)
            elif val_year == 2020:
                month_range = range(1, 7)
            else:
                month_range = range(1, 13)
        elif model == 'NCEP':
            if val_year == 2017:
                month_range = range(7, 10)
            elif val_year == 2019:
                month_range = range(1, 4)
            elif val_year == 2018:
                month_range = range(1, 13)
            else:
                month_range = []
        for val_month in month_range:
            train_index = train_test_index['{}-{:02d}'.format(val_year, val_month)]['train']
            test_index = train_test_index['{}-{:02d}'.format(val_year, val_month)]['test']
            train_X = [cov.loc[train_index].values, subx.loc[idx[:, :, train_index], :].unstack(level=[0, 1]).values]
            test_X = [cov.loc[test_index].values, subx.loc[idx[:, :, test_index], :].unstack(level=[0, 1]).values]
            train_y = truth.loc[idx[:, :, train_index], :].unstack(level=[0, 1]).values
            test_y = truth.loc[idx[:, :, test_index], :].unstack(level=[0, 1]).values
            save_results('forecast/train_y_subx_{}_forecast{}_{}.pkl'.format(val_year, val_month, model), train_y)
            save_results('forecast/test_y_subx_{}_forecast{}_{}.pkl'.format(val_year, val_month, model), test_y)
            save_results('forecast/train_X_subx_{}_forecast{}_{}.pkl'.format(val_year, val_month, model), train_X)
            save_results('forecast/test_X_subx_{}_forecast{}_{}.pkl'.format(val_year, val_month, model), test_X)

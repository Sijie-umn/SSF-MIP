import os
import sys
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import importlib
import pickle
import scipy.io
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from numpy import linalg as LA
from scipy import stats
import sys
from sklearn.metrics import accuracy_score, mean_squared_error
import geopandas as gp
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib
import argparse
import cfg_target
import cfg_target_subx

os.chdir(os.path.join(".."))
sys.path.insert(0, 'SSF_mip/')


def load_results(filename):
    """load a pickle file

    Args:
    filename: the path + file name for the file to be loaded

    Returns: a pickle file
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='EncoderFNN', help='the model used for hyper parameter tuning')
parser.add_argument('--result_type', type=, default='ml', help='the type of results')

args = parser.parse_args()
model_name = args.model_name
lat_lon_grid = pd.read_hdf(cfg_target.absolute_path + 'lat_lon_grid.h5')


if result_type == 'ml':  # ML models
    rootpath = cfg_target.forecast_rootpath + 'forecast_results/'
    result = pd.DataFrame()
    for year in range(2017, 2021):
        if year == 2020:
            month_range = range(1, 7)
        elif year == 2017:
            month_range = range(7, 13)
        else:
            month_range = range(1, 13)
        for month in month_range:
            if month in [1, 3, 5, 7, 8, 10, 12]:
                date = pd.date_range('{}-{:02d}-01'.format(year, month), '{}-{:02d}-31'.format(year, month))
            elif month == 2:
                if year == 2020:
                    date = pd.date_range('{}-{:02d}-01'.format(year, month), '{}-{:02d}-29'.format(year, month))
                else:
                    date = pd.date_range('{}-{:02d}-01'.format(year, month), '{}-{:02d}-28'.format(year, month))
            else:
                date = pd.date_range('{}-{:02d}-01'.format(year, month), '{}-{:02d}-30'.format(year, month))
            res = load_results(rootpath + 'results_{}_{}_{}.pkl'.format(model_name, year, month))
            if model_name == 'EncoderFNN_AllSeq':
                df_one_month = pd.DataFrame(np.stack(res['prediction_test']).mean(axis=0), index=date)
            else:
                df_one_month = pd.DataFrame(res['prediction_test'], index=date)
            result = result.append(df_one_month)
    temp = result.stack().to_frame().reset_index()
    temp.columns = ['start_date', 'lat_lon_index', '{}_fcst'.format(model_name)]
    fcst_results = temp.merge(lat_lon_grid, on='lat_lon_index', how='left').drop('lat_lon_index', axis=1)
    fcst_results = fcst_results.set_index(['lat', 'lon', 'start_date'])
    fcst_results = fcst_results.sort_index()
    fcst_results.to_hdf(rootpath + 'results_{}_{}.h5'.format(model_name, root), key='data')
elif result_type == 'ml_subx':  # ML with subx hindcasts
    rootpath = cfg_target_subx.forecast_rootpath + 'forecast_results/'
    for subx in ['GMAO', 'NCEP', 'wo_GMAO', 'wo_NCEP']:
        if subx in ['GMAO', 'wo_GMAO']:
            subx_date = pd.read_hdf(data_path + 'GMAO_dates.h5')
        else:
            subx_date = pd.read_hdf(data_path + 'NCEP_dates.h5')
        result = pd.DataFrame()
        for year in range(2017, 2021):
            if subx in ['GMAO', 'wo_GMAO']:
                if year == 2020:
                    month_range = range(1, 7)
                elif year == 2017:
                    month_range = range(7, 13)
                else:
                    month_range = range(1, 13)
            else:
                if year == 2020:
                    month_range = []
                elif year == 2017:
                    month_range = range(7, 10)
                elif year == 2019:
                    month_range = range(1, 4)
                else:
                    month_range = range(1, 13)
            for month in month_range:
                date = subx_date[(subx_date.dt.month == month) & (subx_date.dt.year == year)].reset_index(drop=True)
                res = load_results(rootpath + 'results_{}_{}_{}_{}.pkl'.format(model_name, year, month, subx))
                if model_name == 'EncoderFNN_AllSeq':
                    df_one_month = pd.DataFrame(np.stack(res['prediction_test']).mean(axis=0), index=date)
                else:
                    df_one_month = pd.DataFrame(res['prediction_test'], index=date)
                result = result.append(df_one_month)
        temp = result.stack().to_frame().reset_index()
        temp.columns = ['start_date', 'lat_lon_index', '{}_fcst_{}'.format(model_name, subx)]
        fcst_results = temp.merge(lat_lon_grid, on='lat_lon_index', how='left').drop('lat_lon_index', axis=1)
        fcst_results = fcst_results.set_index(['lat', 'lon', 'start_date'])
        fcst_results = fcst_results.sort_index()
        fcst_results.to_hdf(rootpath + 'results_{}_{}.h5'.format(model_name, subx), key='data')

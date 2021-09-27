# compute the spatial/temporal cosine similarity, rmse and relative r2 for each model
import sys
import os
import cfg_target
import cfg_target_subx
import pickle
import numpy as np
from utils import *
import argparse
import evaluation
import pandas as pd
import scipy.io


# post process all results
model_names = cfg_target.model_names
for model_name in model_names:
    if model_name in cfg_target_subx.model_names:
        result_modes = ['ml', 'ml_subx']
    else:
        result_modes = ['ml']
    for result_type in result_modes:
        cmd = "python evaluation/post-processing.py --model_name {}  --result_type {}".format(model_name, result_type)
        print(cmd)
        os.system(cmd)

truth = pd.read_hdf(cfg_target.data_target_file)
rootpath = cfg_target.forecast_rootpath + 'forecast_results/'
idx = pd.IndexSlice
for model_name in model_names:
    result = pd.read_hdf(rootpath + 'results_{}.h5'.format(model_name))
    result = result.merge(truth, on=['lat', 'lon', 'start_date'])
    col = '{}_fcst'.format(model_name)
    target_col = 'target'
    for subx in ['GMAO', 'NCEP']:
        subx_date = pd.read_hdf(cfg_target_subx.subx_data_path + '{}_dates.h5'.format(subx))
        result_subx = result.loc[idx[:, :, subx_date], :]
        temporal_results = pd.DataFrame()
        spatial_results = pd.DataFrame()
        temporal_results[col + '_ACC'] = result_subx.groupby(['lat', 'lon']).apply(lambda df: evaluation.compute_cosine(df[target_col].values, df[col].values))
        temporal_results[col + '_rmse'] = result_subx.groupby(['lat', 'lon']).apply(lambda df: evaluation.compute_rmse(df[target_col].values, df[col].values))
        temporal_results[col + '_r2'] = result_subx.groupby(['lat', 'lon']).apply(lambda df: evaluation.r_squared(df[target_col].values, df[col].values))
        spatial_results[col + '_ACC'] = result_subx.groupby(['start_date']).apply(lambda df: evaluation.compute_cosine(df[target_col].values, df[col].values))
        spatial_results[col + '_rmse'] = result_subx.groupby(['start_date']).apply(lambda df: evaluation.compute_rmse(df[target_col].values, df[col].values))
        spatial_results[col + '_r2'] = result_subx.groupby(['start_date']).apply(lambda df: evaluation.r_squared(df[target_col].values, df[col].values))
        spatial_results.to_hdf(rootpath + 'spatial_results_{}_{}.h5'.format(model_name, subx), key='data')
        temporal_results.to_hdf(rootpath + 'temporal_results_{}_{}.h5'.format(model_name, subx), key='data')
        # to see the stats of the results, use evaluation.print_eval_stats()
        # e.g. evaluation.print_eval_stats(spatial_results[col + '_ACC'])

# merge and evaluate all results - ML & hindcast
rootpath_subx = cfg_target_subx.forecast_rootpath + 'forecast_results/'
for model_name in cfg_target_subx.model_names:
    for subx in ['GMAO', 'NCEP', 'wo_GMAO', 'wo_NCEP']:
        result = pd.read_hdf(rootpath_subx + 'results_{}_{}.h5'.format(model_name, subx))
        result = result.merge(truth, on=['lat', 'lon', 'start_date'])
        col = '{}_fcst_{}'.format(model_name, subx)
        target_col = 'target'
        temporal_results = pd.DataFrame()
        spatial_results = pd.DataFrame()
        temporal_results[col + '_ACC'] = result.groupby(['lat', 'lon']).apply(lambda df: evaluation.compute_cosine(df[target_col].values, df[col].values))
        temporal_results[col + '_rmse'] = result.groupby(['lat', 'lon']).apply(lambda df: evaluation.compute_rmse(df[target_col].values, df[col].values))
        temporal_results[col + '_r2'] = result.groupby(['lat', 'lon']).apply(lambda df: evaluation.r_squared(df[target_col].values, df[col].values))
        spatial_results[col + '_ACC'] = result.groupby(['start_date']).apply(lambda df: evaluation.compute_cosine(df[target_col].values, df[col].values))
        spatial_results[col + '_rmse'] = result.groupby(['start_date']).apply(lambda df: evaluation.compute_rmse(df[target_col].values, df[col].values))
        spatial_results[col + '_r2'] = result.groupby(['start_date']).apply(lambda df: evaluation.r_squared(df[target_col].values, df[col].values))
        spatial_results.to_hdf(rootpath_subx + 'spatial_results_{}_{}.h5'.format(model_name, subx), key='data')
        temporal_results.to_hdf(rootpath_subx + 'temporal_results_{}_{}.h5'.format(model_name, subx), key='data')
        # to see the stats of the results, use evaluation.print_eval_stats()
        # e.g. evaluation.print_eval_stats(spatial_results[col + '_ACC'])

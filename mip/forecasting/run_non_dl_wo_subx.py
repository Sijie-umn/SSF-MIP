import sys
import os
os.chdir(os.path.join(".."))
sys.path.insert(0, 'SSF_mip/')
import numpy as np
import pandas as pd
import cfg_target_subx as cfg_target
import pickle
from random import randint
from random import seed
import torch
import model
from joblib import Parallel, delayed
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='model_name')
parser.add_argument('--year', type=int, help='year')
# parser.add_argument('--month', type=int, help='month')
# parser.add_argument('--subx_model', type=str, default='GMAO', help='the model used for hyper parameter tuning')

args = parser.parse_args()
model_name = args.model_name
year = args.year
# month_id = args.month
# subx_model = args.subx_model


def forecast_non_dl(month_id, year, rootpath, param_path, model_name, subx_model):
    """Run non deep learning models (XGBoost and Multitask Lasso) - results are saved in a folder named forecast_results
    Args:
    month_id: an int indicating the month which is being forecasted
    year: an int indicating the year which is being forecasted
    rootpath: the path where the training and test sets are saved
    param_path: the path where the best hyperparameters are saved
    device: an indication if the model is runing on GPU or CPU
    model_name: a string indicating the name of a model
    one_day: True or False, indicating if only the most recent available day is used for training a model (XGBoost or Lasso)
    """
    results = {}
    train_X = load_results(rootpath + 'train_X_subx_{}_forecast{}_{}.pkl'.format(year, month_id, subx_model))
    test_X = load_results(rootpath + 'test_X_subx_{}_forecast{}_{}.pkl'.format(year, month_id, subx_model))
    train_y = load_results(rootpath + 'train_y_subx_{}_forecast{}_{}.pkl'.format(year, month_id, subx_model))
    test_y = load_results(rootpath + 'test_y_subx_{}_forecast{}_{}.pkl'.format(year, month_id, subx_model))

    input_dim = train_X[0].shape[-1]
    output_dim = train_y.shape[-1]
    # ar_dim = train_X.shape[1]
    best_parameter = load_results(param_path + '{}_forecast{}_{}.pkl'.format(model_name, month_id, subx_model))
    if model_name == 'XGBoost':
        curr_max_depth = best_parameter['max_depth']
        curr_colsample_bytree = best_parameter['colsample_bytree']
        curr_gamma = best_parameter['gamma']
        curr_n_estimators = best_parameter['n_estimators']
        curr_lr = best_parameter['learning_rate']
        mdl = model.XGBMultitask_wo_subx(num_models=output_dim, colsample_bytree=curr_colsample_bytree,
                                         gamma=curr_gamma, learning_rate=curr_lr, max_depth=curr_max_depth,
                                         n_estimators=curr_n_estimators, objective='reg:squarederror')

    elif model_name == 'Lasso':
        curr_alpha = best_parameter['alpha']
        mdl = model.LassoMultitask_wo_subx(alpha=curr_alpha, num_models=output_dim, n_jobs=16, fit_intercept=False)

    mdl.fit(train_X, train_y)
    # send model to gpu
    filename = rootpath + 'forecast_results/models/model_{}_{}_{}_wo_{}.sav'.format(model_name, year, month_id, subx_model)
    pickle.dump(mdl, open(filename, 'wb'))
    pred_train = mdl.predict(train_X)
    pred_test = mdl.predict(test_X)
    results['target_train'] = train_y
    results['prediction_train'] = pred_train
    results['target_test'] = test_y
    results['prediction_test'] = pred_test
    save_results(rootpath + 'forecast_results/results_{}_{}_{}_wo_{}.pkl'.format(model_name, year, month_id, subx_model), results)


# set device for running the code
rootpath = cfg_target.forecast_rootpath
param_path = cfg_target.param_path
# one_day = cfg_target.one_day
for subx_model in ['GMAO']:
    if year == 2017:
        month_range = range(7, 13)
    elif year == 2020:
        month_range = range(1, 7)
    else:
        month_range = range(1, 13)
    for month_id in month_range:
        forecast_non_dl(month_id=month_id, year=year, rootpath=rootpath, param_path=param_path, model_name=model_name, subx_model=subx_model)

for subx_model in ['NCEP']:
    if year == 2017:
        month_range = range(7, 10)
    elif year == 2019:
        month_range = range(1, 4)
    elif year == 2018:
        month_range = range(1, 13)
    else:
        month_range = []
    for month_id in month_range:
        forecast_non_dl(month_id=month_id, year=year, rootpath=rootpath, param_path=param_path, model_name=model_name, subx_model=subx_model)

# Parallel(n_jobs=12)(delayed(forecast_rep)(month_id, rootpath=rootpath, param_path=param_path, device=device, model_name=model_name, folder_name=folder_name, num_rep= num_rep) for month_id in range(1,13))

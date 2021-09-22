#!/usr/bin/env python
# coding: utf-8

# need a function for clarify the spatial range
import os
import sys
from random import randint
from random import seed
os.chdir(os.path.join(".."))
sys.path.insert(0, 'SSF_mip/')
from utils import *
import cfg_target_subx as cfg_target
import torch
import model
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import math
import argparse
import numpy as np
import pandas as pd
import math
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import joblib
import argparse
import torch
import torch.nn as nn


def compute_cosine(a, b):
    """Compute cosine similarity between two vectors
    Args:
    a,b: numpy array
    Returns: a float (cosine similarity)
    """
    return np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))


def random_cv(cv_index, cv_year, roothpath, param_grid, num_random, model_name, device, subx_model):
    """Hyperparameter tuning through random search

    Args:
    cv_index: the month of the valiation set
    cv_year: the year of the valiation set
    rootpath: the path where training-validtion sets are saved
    param_grid: a dictionary, consisting the grid of hyperparameters
    num_randon: the number of sets of hyperparameters to evaluate(tune)
    model_name: a string representing the name of a model
    device: indicates if the model should be run on cpu or gpu
    one_day: True or False, indicating if only the most recent available day is used for training a model (XGBoost or Lasso)
    """
    # load data
    train_X = load_results(rootpath + 'train_X_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
    valid_X = load_results(rootpath + 'val_X_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
    train_y = load_results(rootpath + 'train_y_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
    valid_y = load_results(rootpath + 'val_y_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
    # set input and output dim
    input_dim = train_X.shape[-1]
    output_dim = train_y.shape[-1]

    if model_name == 'XGBoost':
        max_depth = param_grid['max_depth']
        colsample_bytree = param_grid['colsample_bytree']
        gamma = param_grid['gamma']
        n_estimators = param_grid['n_estimators']
        lr = param_grid['learning_rate']
    elif model_name == 'Lasso':
        alphas = param_grid['alpha']
    else:
        print('the model name is not in the list')

    history_all = []
    score = []
    parameter_all = []
    for i in range(num_random):
        if model_name == 'XGBoost':
            curr_max_depth = max_depth[randint(0, len(max_depth) - 1)]
            curr_colsample_bytree = colsample_bytree[randint(0, len(colsample_bytree) - 1)]
            curr_gamma = gamma[randint(0, len(gamma) - 1)]
            curr_n_estimators = n_estimators[randint(0, len(n_estimators) - 1)]
            curr_lr = lr[randint(0, len(lr) - 1)]
            parameters = {'max_depth': curr_max_depth, 'colsample_bytree': curr_colsample_bytree,
                          'gamma': curr_gamma, 'n_estimators': curr_n_estimators,
                          'learning_rate': curr_lr}
            parameter_all.append(parameters)
            mdl = model.XGBMultitask_wo_subx(num_models=output_dim, colsample_bytree=curr_colsample_bytree,
                                             gamma=curr_gamma, learning_rate=curr_lr, max_depth=curr_max_depth,
                                             n_estimators=curr_n_estimators, objective='reg:squarederror')
            # history = mdl.fit_cv(train_X, train_y, valid_X, valid_y)
            mdl.fit(train_X, train_y)
            pred_y = mdl.predict(valid_X)
            history = None
        elif model_name == 'Lasso':
            curr_alpha = alphas[randint(0, len(alphas) - 1)]
            parameter = {'alpha': curr_alpha}
            parameter_all.append(parameter)
            mdl = model.LassoMultitask_wo_subx(alpha=curr_alpha, num_models=output_dim, n_jobs=16, fit_intercept=False)
            mdl.fit(train_X, train_y)
            pred_y = mdl.predict(valid_X)
            history = None

        history_all.append(history)
        test_rmse = np.sqrt(((valid_y - pred_y)**2).mean())
        test_cos = np.asarray([compute_cosine(valid_y[i, :], pred_y[i, :]) for i in range(len(valid_y))]).mean()
        score.append([test_rmse, test_cos])

    cv_results = {'score': score, 'parameter_all': parameter_all, 'history_all': history_all}
    save_results(rootpath + 'cv_results_test/cv_results_' + model_name + '_{}_{}_wo_{}.pkl'.format(cv_year, cv_index, subx_model), cv_results)


# set device for running the code

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set the seed to gauarantee for each validation set the hyperparameters are shown in the same sequence
seed(314)

# param_grid = cfg_target.param_grid_en_de
month_range = cfg_target.month_range
val_years = cfg_target.val_years
num_random = cfg_target.num_random
rootpath = cfg_target.rootpath_cv

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2011, help='the year selected for hyper parameter tuning')
parser.add_argument('--model_name', type=str, default='EncoderFNN', help='the model used for hyper parameter tuning')
parser.add_argument('--subx_model', type=str, default='GMAO', help='the model used for hyper parameter tuning')

args = parser.parse_args()
year = args.year
model_name = args.model_name
subx_model = args.subx_model
# print(model_name)
if model_name == 'XGBoost':
    param_grid = cfg_target.param_grid_xgb
elif model_name == 'Lasso':
    param_grid = cfg_target.param_grid_lasso
elif model_name == 'EncoderFNN_AllSeq':
    param_grid = cfg_target.param_grid_en_de
else:
    print('can not find the model')

Parallel(n_jobs=12)(delayed(random_cv)(cv_index, cv_year=year, roothpath=rootpath, param_grid=param_grid, num_random=num_random, model_name=model_name, device=device, subx_model=subx_model) for cv_index in month_range)

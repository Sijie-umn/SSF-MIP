"""
Run random_cv.py (in the folder hyperparameter_tuning) for hyper parameter tuning
"""
import os
import cfg_target_subx as cfg_target
import pickle
import numpy as np
from utils import *


model_names = cfg_target.model_names
val_years = cfg_target.val_years
month_range = cfg_target.month_range
rootpath = cfg_target.rootpath_cv
metric = cfg_target.cv_metric


def best_hyperparameter_subx(val_years, month_range, eval_metrics, model_name, rootpath, subx_model, subx):
    """Find the best hyper parameters based on the results on validation set

    Args:
    val_years: an array with the years considered for validation set
    month_range: an array with the months for hyperparameter tuning
    eval_metrics: a string indicating the evaluation metric ('cos' or 'rmse')
    model_name: a string, the name of the model for hyperparameter tuning
    rootpath: the path where the validation results are saved
    """
    for month in month_range:
        score_all = []
        for year in val_years:
            # print(month, year)
            if subx is True:
                cv_results = load_results(rootpath + 'cv_results_test/cv_results_' + model_name + '_{}_{}_{}.pkl'.format(year, month, subx_model))
            else:
                cv_results = load_results(rootpath + 'cv_results_test/cv_results_' + model_name + '_{}_{}_wo_{}.pkl'.format(year, month, subx_model))
            score = cv_results['score']
            score_all.append(score)
        score_all = np.asarray(score_all).squeeze()

        if eval_metrics == 'cos':
            best_score = score_all[:, :, 1].mean(axis=0)
            best_id = np.where(best_score == best_score.max())
        elif eval_metrics == 'rmse':
            best_score = score_all[:, :, 0].mean(axis=0)
            best_id = np.where(best_score == best_score.min())

        best_parameter = cv_results['parameter_all'][best_id[0][0]]
        if subx is True:
            save_results(rootpath + 'cv_results_test/best_parameter/{}_forecast{}_{}.pkl'.format(model_name, month, subx_model), best_parameter)
        else:
            save_results(rootpath + 'cv_results_test/best_parameter/{}_forecast{}_wo_{}.pkl'.format(model_name, month, subx_model), best_parameter)


for model_name in model_names:
    for subx_model in ['GMAO', 'NCEP']:
        for year in val_years:
            cmd = "{} {} --year {} --model_name {} --subx_model {}".format("python", 'hyperparameter_tuning/random_cv_subx.py', year, model_name, subx_model)
            print(cmd)
            os.system(cmd)
            cmd = "{} {} --year {} --model_name {} --subx_model {}".format("python", 'hyperparameter_tuning/random_cv_wo_subx.py', year, model_name, subx_model)
            print(cmd)
            os.system(cmd)
    print('find the best hyper parameter for {}'.format(model_name))
    best_hyperparameter_subx(val_years=val_years, month_range=month_range, eval_metrics=metric, model_name=model_name, rootpath=rootpath, subx_model=subx_model, subx=True)
    best_hyperparameter_subx(val_years=val_years, month_range=month_range, eval_metrics=metric, model_name=model_name, rootpath=rootpath, subx_model=subx_model, subx=False)

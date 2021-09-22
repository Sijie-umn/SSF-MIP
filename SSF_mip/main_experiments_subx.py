"""
Run models in config file for test set with the best hyperparameter selected based on the validation sets
"""
import os
import cfg_target_subx as cfg_target
import pickle
import numpy as np
from utils import *
import argparse
import evaluation

test_years = cfg_target.test_years
rootpath = cfg_target.forecast_rootpath


for model_name in ['XGBoost', 'Lasso']:
    for subx in [True, False]:
        if subx is True:
            file_name = 'forecasting/run_non_dl_subx.py'
        else:
            file_name = 'forecasting/run_non_dl_wo_subx.py'
    for year in test_years:
        cmd = "{} {} --model_name {} --year {}".format("python", file_name, model_name, year)
        print(cmd)
        os.system(cmd)

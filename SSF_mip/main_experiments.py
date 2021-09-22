"""
Run models in config file for test set with the best hyperparameter selected based on the validation sets
"""
import os
import cfg_target
import pickle
import numpy as np
from utils import *
import argparse
import evaluation

model_names = cfg_target.model_names
test_years = cfg_target.test_years
# month_range = cfg_target.month_range
rootpath = cfg_target.forecast_rootpath


for model_name in model_names:
    if model_name == 'EncoderFNN_AllSeq':
        file_name = 'forecasting/run_encoder_decoder.py'
    elif model_name in ['XGBoost', 'Lasso']:
        file_name = 'forecasting/run_non_dl.py'
    for year in test_years:
        if year == 2020:
            month_range = range(1, 7)
        elif year == 2017:
            month_range = range(7, 13)
        else:
            month_range = range(1, 13)
        for month in month_range:
            cmd = "{} {} --model_name {} --year {} --month {}".format("python", file_name, model_name, year, month)
            print(cmd)
            os.system(cmd)

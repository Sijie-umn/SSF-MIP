import sys
import os
os.chdir(os.path.join(".."))
sys.path.insert(0, 'SSF_mip/')
import numpy as np
import pandas as pd
import cfg_target as cfg_target
import pickle
from random import randint
from random import seed
import torch
import model
from joblib import Parallel, delayed
from utils import *
# from sklearn.linear_model import LogisticRegression
import argparse
import math
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import argparse
import torch
import torch.nn as nn


# parser = argparse.ArgumentParser()
# parser.add_argument('--model_name', type=str, help='model_name')
# parser.add_argument('--year', type=int, help='year')
# parser.add_argument('--region', type=str, help='region_name')
# parser.add_argument('--testing', type=str, help='trick')
# parser.add_argument('--loss_type', type=str, help='loss_type')

# args = parser.parse_args()
# model_name = args.model_name
# year = args.year
# region = args.region
# month = args.month
# testing = args.testing
# loss_type = args.loss_type


def forecast_rep(month_id, year, rootpath, device, model_name, num_rep):
    """Run encoder-decoder style models with repetition - results are saved in a folder named forecast_results
    Args:
    month_id: an int indicating the month which is being forecasted
    year: an int indicating the year which is being forecasted
    rootpath: the path where the training and test sets are saved
    param_path: the path where the best hyperparameters are saved
    device: an indication if the model is runing on GPU or CPU
    model_name: a string indicating the name of a model
    num_rep: the number of repetition
    """
    results = {}
    results['prediction_train'] = []
    results['prediction_test'] = []
    results['history'] = []
    train_X = load_results(rootpath + 'train_X_pca_{}_forecast{}.pkl'.format(year, month_id))
    test_X = load_results(rootpath + 'test_X_pca_{}_forecast{}.pkl'.format(year, month_id))
    train_y = load_results(rootpath + 'train_y_pca_{}_forecast{}.pkl'.format(year, month_id))
    test_y = load_results(rootpath + 'test_y_pca_{}_forecast{}.pkl'.format(year, month_id))

    input_dim = train_X.shape[-1]
    output_dim = train_y.shape[-1]
    # ar_dim = train_X.shape[1]
    # best_parameter = load_results(param_path + '{}_forecast{}.pkl'.format(model_name, month_id))
    for rep in range(num_rep):
        curr_hidden_dim = 64
        curr_num_layer = 2
        curr_seq_len = 18
        curr_threshold = 2.5
        curr_lr = 0.00001
        curr_num_epochs = 200
        curr_linear_dim = 32
        curr_drop_out = 0.2

        mdl = model.EncoderFNN_AllSeq(input_dim=input_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim,
                                      num_layers=curr_num_layer, seq_len=curr_seq_len, linear_dim=curr_linear_dim,
                                      learning_rate=curr_lr, dropout=curr_drop_out, threshold=curr_threshold,
                                      num_epochs=curr_num_epochs)
        # set data for training
        train_dataset = model.MapDataset(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
        model.init_weight(mdl)
        # send model to gpu
        mdl.to(device)
        history = mdl.fit_cv(train_loader, test_X, test_y, device)
        pred_train = mdl.predict(train_X, device)
        pred_y = mdl.predict(test_X, device)

        results['target_train'] = train_y
        results['prediction_train'].append(pred_train)
        results['target_test'] = test_y
        results['prediction_test'].append(pred_y)
        results['history'].append(history)
    save_path = '/glade/work/hexxx/SSF_mip/model_testing/'
    save_results(save_path + 'results_{}_{}_{}_{}.pkl'.format(model_name, year, month_id, testing), results)


model_name = 'EncoderFNN_AllSeq'
testing = 'ld-32-lr-5'
year = 2017
rootpath = '/glade/work/hexxx/SSF_mip/data_us/forecast/'
num_rep = 2
# set device for running the code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Parallel(n_jobs=-2)(delayed(forecast_rep)(month_id, year=year, rootpath=rootpath, device=device, model_name=model_name, num_rep=num_rep) for month_id in range(7, 13))
year = 2020
Parallel(n_jobs=-2)(delayed(forecast_rep)(month_id, year=year, rootpath=rootpath, device=device, model_name=model_name, num_rep=num_rep) for month_id in range(1, 7))

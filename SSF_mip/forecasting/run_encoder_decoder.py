import numpy as np
import pandas as pd
import sys
import os
os.chdir(os.path.join(".."))
sys.path.insert(0, 'SSF_mip/')
import cfg_target
import pickle
from random import randint
from random import seed
import torch
import model
import argparse
from joblib import Parallel, delayed
from utils import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='model_name')
parser.add_argument('--year', type=int, help='year')
parser.add_argument('--month', type=int, help='month')

args = parser.parse_args()
model_name = args.model_name
year = args.year
month_id = args.month


def forecast_rep(month_id, year, rootpath, param_path, device, model_name, num_rep):
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
    train_X = load_results(rootpath + 'train_X_pca_{}_forecast{}.pkl'.format(year, month_id))
    test_X = load_results(rootpath + 'test_X_pca_{}_forecast{}.pkl'.format(year, month_id))
    train_y = load_results(rootpath + 'train_y_pca_{}_forecast{}.pkl'.format(year, month_id))
    test_y = load_results(rootpath + 'test_y_pca_{}_forecast{}.pkl'.format(year, month_id))

    input_dim = train_X.shape[-1]
    output_dim = train_y.shape[-1]
    # ar_dim = train_X.shape[1]
    best_parameter = load_results(param_path + '{}_forecast{}.pkl'.format(model_name, month_id))
    for rep in range(num_rep):
        if model_name == 'EncoderFNN_AllSeq':
            curr_hidden_dim = best_parameter['hidden_dim']
            curr_num_layer = best_parameter['num_layers']
            curr_seq_len = best_parameter['seq_len']
            curr_threshold = best_parameter['threshold']
            curr_lr = best_parameter['learning_rate']
            curr_num_epochs = best_parameter['num_epochs']
            curr_linear_dim = best_parameter['linear_dim']
            curr_drop_out = best_parameter['drop_out']

            mdl = model.EncoderFNN_AllSeq(input_dim=input_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim,
                                          num_layers=curr_num_layer, seq_len=curr_seq_len, linear_dim=curr_linear_dim,
                                          learning_rate=curr_lr, dropout=curr_drop_out, threshold=curr_threshold,
                                          num_epochs=curr_num_epochs)
            # set data for training
            train_dataset = model.MapDataset(train_X, train_y)
            train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
            model.init_weight(mdl)
            # send model to gpu
            mdl.to(device)
            mdl.fit(train_loader, device)
            state = {'state_dict': mdl.state_dict()}
            torch.save(state, rootpath + 'models/{}_{}_{}.t7'.format(model_name, year, month_id))
            pred_train = mdl.predict(train_X, device)
            pred_y = mdl.predict(test_X, device)

        results['target_train'] = train_y
        results['prediction_train'].append(pred_train)
        results['target_test'] = test_y
        results['prediction_test'].append(pred_y)

    save_results(rootpath + 'forecast_results/results_{}_{}_{}.pkl'.format(model_name, year, month_id), results)


# set device for running the code
num_rep = cfg_target.num_rep
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rootpath = cfg_target.forecast_rootpath
param_path = cfg_target.param_path

forecast_rep(month_id=month_id, year=year, rootpath=rootpath, param_path=param_path, device=device, model_name=model_name, num_rep=num_rep)
# Parallel(n_jobs=12)(delayed(forecast_rep)(month_id,rootpath=rootpath,param_path=param_path, device= device, model_name=model_name,folder_name=folder_name, num_rep= num_rep) for month_id in range(1,13))

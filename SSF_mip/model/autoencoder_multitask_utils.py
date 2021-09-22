from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np


class MapDataset(Dataset):
    """
    X: of size [num_samples, num_features], is the covariates matrix
    y: of size [num_samples,num_locations], is the target variable, which has the same length as each element in X
    """
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.labels)  # of how many examples(images?) you have

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def init_weight(mdl):
    """Initalize the weights for each deep learning model
    Args:
    mdl: a deep learning model
    """
    for name, param in mdl.named_parameters():
        if 'weight' in name:
            param = nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))

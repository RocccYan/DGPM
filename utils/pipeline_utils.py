# -*- coding: utf-8 -*-
import argparse
import copy
from collections import defaultdict
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
import torch
from torch import nn
import torch_geometric.transforms as T
from torch_geometric import datasets
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

import sys
sys.path.append("/mnt/workspace/graph_pretrain")
from MotiFiesta.utils.split_data import split_dataset
from MotiFiesta.pretrain.models import GCNEncoder,GINEncoder
from MotiFiesta.src.model import MotiFiesta
# from MotiFiesta.pretrain.graphmae.models import build_model


def mkdir(folder_path):
    try:
        os.mkdir(folder_path)
        print(f"******** make dir {folder_path} *************")
    except Exception as e:
        # print(e)
        pass


def get_splits(dataset, fold_idx):
    idxpath = '../supervised_data/fold-idx/{}/{}_idx-{}.txt'

    test_idx = np.loadtxt(idxpath.format(dataset, 'test', fold_idx), dtype=int)
    train_idx = np.loadtxt(idxpath.format(dataset, 'train', fold_idx), dtype=int)
    size = len(train_idx)
    val_size = size // 9
    val_idx = train_idx[-val_size:]
    train_idx = train_idx[:-val_size]
    return train_idx, val_idx, test_idx


def split_dataset(labels,dataset_name,n_splits=10,seed=42):
    # check first
    fold_idx_path = f'../supervised_data/fold-idx/{dataset_name}/'
    if os.path.exists(fold_idx_path) and os.listdir(fold_idx_path):
        return 
    else:
        os.makedirs(fold_idx_path, exist_ok=True)

    idxpath = '../supervised_data/fold-idx/{}/{}_idx-{}.txt'
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx_list = []
    for fold_idx, idx in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # idx_list.append(idx)
        train_idx, test_idx = idx
        np.savetxt(idxpath.format(dataset_name, 'train', fold_idx),train_idx,delimiter=',',fmt='%d')
        np.savetxt(idxpath.format(dataset_name, 'test', fold_idx),test_idx,delimiter=',',fmt='%d')


def read_json(filepath):
    value = None
    try:
        with open(filepath,'r') as fh:
            value = json.load(fh)
    except Exception as e:
        print(f"Json read failed: {e}")
    return value


def save_as_json(obj, filepath):
    try:
        with open(filepath,'w') as fh:
            json.dump(obj, fh, 
                ensure_ascii=True, indent=4, allow_nan=True)
    except Exception as e:
        print(f"Json save failed: {e}")


def load_pretrained_model(model_path,
                          model_name,
                          input_channels,
                          hidden_channels,
                          num_layers):
    r"""parse model_path first, and build a model, then load the state_dict().
        Args:
            model_path(str): ../runs/COX2_GIN_B=512_lr=0.001/model_E=100.pt
    """
    if model_name == 'GIN':
        encoder = GINEncoder
    else:
        encoder = GCNEncoder
    model = encoder(input_channels, hidden_channels, num_layers)
    model.load_state_dict(torch.load(model_path))
    return model


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()}. ", end='\t')
        print(f"device {torch.cuda.get_device_name(0)} will be used.")
    else:
        device = torch.device('cpu')
        print('No GPU is available, cpu will be used instead.')
    return device


def set_workspace(path='.'):

    if os.path.exists(path):
        os.chdir(path)
        print(f'workspace: {os.getcwd()}')


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
        return self.early_stop
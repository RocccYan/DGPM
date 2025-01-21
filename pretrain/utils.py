#! usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import torch
import random
import torch
import numpy as np
import argparse

from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F

import sys
sys.path.append("/mnt/workspace/graph_pretrain/MotiFiesta/pretrain")
from graphmae.models import build_model
from MotiFiesta.src.model import MotiFiesta


def mkdir(folder_path):
    try:
        os.mkdir(folder_path)
    except Exception as e:
        # print(e)
        pass


def save_as_json(data, filename, mode='w'):
    with open(filename, mode, encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=4,
                #   use_decimal=True
                  )


def open_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    return data


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


def label_distribution(dataset,num_classes=2):
    # num_classes = dataset.num_classes
    classes_distribution = dict()
    for i in range(num_classes):
        if isinstance(dataset[0], dict):
            ratio = round(sum(list(map(lambda x: x['label']==i, dataset))).numpy() / len(dataset), 6)
        else:
            ratio = round(sum(list(map(lambda x: x[1]==i, dataset))).numpy() / len(dataset), 6)
        classes_distribution.update({i:ratio})
    # print(f'classes distribution: {classes_distribution}')
    return classes_distribution


def get_item_by_list(a,b):
    return list(map(lambda idx: a[idx], b))

def train_test_split(X, y=None, train_size=None, test_size=None, seed=42):
    length = len(X)
    index = list(range(length))
    random.Random(seed).shuffle(index)
    if train_size is None:
        if test_size is not None:
            train_size = 1 - test_size
        else:
            train_size, test_size = 0.8, 0.2
    num_train_samples, num_test_samples = int(train_size * length), int(test_size * length)
    # take test samples from the last one backwardly.
    X_train, X_test = get_item_by_list(X, index[:num_train_samples]), get_item_by_list(X, index[num_test_samples*(-1):])
    if y is not None:
        y_train, y_test = get_item_by_list(y, index[:num_train_samples]), get_item_by_list(y, index[num_test_samples*(-1):])
        return X_train, X_test, y_train, y_test
    
    return X_train, X_test


def kf_split_fewshot(X, y=None, fewshot_ratio=0.1, n_splits=5, random_state=42):
    kf_split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fewshot_size = int(fewshot_ratio * len(X))
    fewshot_splits = list(map(lambda split: (np.random.RandomState(seed=random_state).choice(split[0], fewshot_size), split[1]), kf_split.split(X,y)))
    return fewshot_splits


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


def load_saved_model(model_name, model_type, model_file_name="", root='./models',load_optimizer=False, load_cross_trained=False):
    print("*"*20)
    print(os.getcwd())
    # load model hparams
    hparams = json.load(open(os.path.join(root,model_name,'hparams.json')))
    if model_type == 'node':
        # model = Model(**hparams)
        args = argparse.Namespace(**hparams)
        model = build_model(args)
        if model_file_name:
            model_file = model_file_name
        else:
            model_file = 'model_node.pth' if load_cross_trained else 'model.pth'
        params = torch.load(
            os.path.join(root,model_name,model_file),)
    elif model_type == 'motif':
        model = MotiFiesta(**hparams['model'])
        # load model params
        if load_cross_trained:
            model_file_name = model_file_name if model_file_name else "model_motif.pth"
            params = torch.load(
                os.path.join(root,model_name,model_file_name),map_location='cpu')
        else:
            model_file_name = model_file_name if model_file_name else f"{model_name}.pth"
            params = torch.load(
                os.path.join(root,model_name,model_file_name),map_location='cpu')['model_state_dict']
    params = {k.replace('module.',''):v for k,v in params.items()}
    model.load_state_dict(params)
    if load_optimizer:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(torch.load(
            os.path.join(root,model_file,'optimizer.pth'),)
        )
        return model, optimizer
    else:
        return model

def motifID2labels(motifs):
    motifs_set = sorted(set(motifs.numpy()))
    motif2label_map = {motif:idx for idx,motif in enumerate(motifs_set)}
    labels = torch.tensor([motif2label_map[motif] for motif in motifs.numpy()],device=motifs.device)
    return labels

def spotlight2label(spotlights, mode='cat'):
    # check whether spotlight id from 0
    labels_dict = {}
    for spotlight_id, nodes in spotlights.items():
        for node in nodes:
            labels_dict[node] = spotlight_id
    labels = torch.tensor([x[1] for x in sorted(labels_dict.items())])
    if mode == 'cat':
        # onehot and flatten
        labels = F.one_hot(labels, len(spotlights)).flatten()
    return labels

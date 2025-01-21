# -*- coding: utf-8 -*-


"""### load"""
import os
import json
import random
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
from abc import ABC
from functools import partial
import copy as cp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d,Sequential, Linear, ReLU
from torch.optim import AdamW

from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool



def matrix_similarity(a,b):
    """a, b are batches of embs with same length."""
    a_norm = a.norm(dim=1)
    b_norm = b.norm(dim=1)

    matrix_sim = torch.einsum('ik,jk->ij', a, b) / torch.einsum('i,j->ij', a_norm, b_norm)
    return matrix_sim

def matching_loss(a, b, label):
    matrix_sim = matrix_similarity(a, b)
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(matrix_sim, label)

def cat_matching_loss(a, b, label):
    """cat a (m1,n1) and b(m2,n2) into (m1*m2, n1+n2)"""
    a = torch.unsqueeze(a, 1).repeat(1, b.shape[0], 1)
    b = torch.unsqueeze(b, 1).repeat(1, a.shape[0], 1)
    c = torch.cat([a.view(-1,a.shape[-1]),b.view(-1, b.shape[-1])],dim=1)
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(c, label)


 
"""# gcl loss"""

def gcl_loss(x1, x2, temperature=0.5):
    temperature = 0.5
    batch_size, _ = x1.size()

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    
    return loss


"""### Model"""

def readout(x, batch, ptr, mode='global_mean'):
    if mode == 'global_add':
        x = global_add_pool(x, batch)
    elif mode == 'global_mean':
        x = global_mean_pool(x, batch)
    # [batch_size, hidden_channels]
    elif mode == 'all':
        x = torch.cat(tuple([
            global_add_pool(x, batch),
            global_max_pool(x, batch),
            global_mean_pool(x, batch),
            ]),
            dim = 1)
    return x
    
# classifier
class GNNClassifier(torch.nn.Module):
    def __init__(self, encoder, num_features, num_classes):
        super(GNNClassifier, self).__init__()

        self.encoder = encoder
        self.lin = Linear(num_features, num_classes)

    def forward(self, data):
        x = self.encoder(data)
        x = self.lin(x)

        return x

# Encoders

# GCN
class GCNEncoder(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=2, readout='all'):
        super().__init__()

        self.readout = readout

        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        if self.readout == 'all':
            num_pooler = 3
        else:
            num_pooler = 1

        self.projection_head = Sequential(
            Linear(hidden_channels*num_pooler, hidden_channels*num_pooler), 
            ReLU(inplace=True), 
            Linear(hidden_channels*num_pooler, hidden_channels*num_pooler)
            )
    
    def forward(self, data):
        x, edge_index, batch, ptr = data.x, data.edge_index, data.batch, data.ptr
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        
        x = readout(x,batch,ptr,mode=self.readout)
        x = F.dropout(x, p=0.2, training=self.training)
        
        return x
    
    def forward_cl(self, data):
        x = self.forward(data)
        x = self.projection_head(x)
        return x




class GINEncoder(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels=32, num_layers=3):
        super(GINEncoder, self).__init__()

        self.num_gc_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):
            if i:
                nn = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
            else:
                nn = Sequential(Linear(input_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(hidden_channels)

            self.convs.append(conv)
            self.bns.append(bn)

        self.projection_head = Sequential(
            Linear(hidden_channels*num_layers, hidden_channels*num_layers), 
            ReLU(inplace=True), 
            Linear(hidden_channels*num_layers, hidden_channels*num_layers)
            )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)

        return x #, torch.cat(xs, 1)

    def forward_cl(self, data):
        x = self.forward(data)
        x = self.projection_head(x)
        return x

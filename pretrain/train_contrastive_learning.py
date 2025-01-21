# -*- coding: utf-8 -*-
"""
This class aims to take subgraphs generated according motifs as 
positive/negative views, and do contrastive learning, which return
a pretrained graph encoder. Pipeline 
is informed by <https://openreview.net/forum?id=qcKh_Msv1GP>, 
Motif Driven Graph Representation Learning by Yizhou Sun in 2021,
consist of Generating views and load batchs by motif label, encoding,
and computing loss and backpropagation.

Args: 
    dataset (pyg Dataset): dataset of graphs with node features 
        and motif label/distribution.
    graph_encoder (GNN model): encode subgraph as embeddings
"""

from collections import defaultdict
import json
import os
import time

import numpy as np
import pandas as pd
import random
import torch 
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from build_dataset import MotifDataset, shuffle_with_fixed_motif_order
from configs import DataArguments, ModelArguments, CLTrainArguments, HfArgumentParser
from models import GINEncoder, GCNEncoder, BiGCNEncoder,ResGCNEncoder, gcl_loss
from utils import set_device, set_workspace, mkdir

# TODO: set workspace
 
def pipeline(dataset, encoder, data_args, model_args, training_args, ):
    
    # shuffle data with motif
    dataset = dataset.shuffle()
    dataset_shuffled = shuffle_with_fixed_motif_order(dataset)

    dataloader_1 = DataLoader(dataset=dataset, batch_size=training_args.batch_size)
    dataloader_2 = DataLoader(dataset=dataset_shuffled, batch_size=training_args.batch_size)

    # TODO: load exist task and continue training...
    model = encoder(model_args.input_channels, model_args.hidden_channels, model_args.num_layers).to(device)
    optimizer = AdamW(model.parameters(), lr=training_args.lr)
    criterion = gcl_loss

    model_file_name = f"{training_args.task_name}={model_args.model_name}={model_args.num_layers}={data_args.dataset_name}={training_args.batch_size}={training_args.lr}={model_args.eta}={training_args.temperature}"
    for epoch in range(1, training_args.num_epochs+1):
        logs = defaultdict(list)
        train_loss = train(dataloader_1, dataloader_2, model, optimizer, criterion, device, model_args, training_args)
        logs['epoch'].append(epoch)
        logs['train loss'].append(train_loss) 
        # TODO: save model with hparams.
        if epoch % training_args.print_per_epoch == 0:
            torch.save(model.state_dict(), f"./../runs/{training_args.task_name}/model_E={epoch}.pt")
            print(f'[INFO]model saved at epoch {epoch}.')

    pd.DataFrame.from_dict(logs).to_csv(f"./../runs/{training_args.task_name}/log.csv")
    print("Contrastive Learning Finished!")


def train(dataloader_1, dataloader_2, model, optimizer, criterion, device, model_args, training_args):
    model.train()
    total_loss = 0
    for data_1, data_2 in tqdm(zip(dataloader_1, dataloader_2), desc='Training...'):
        optimizer.zero_grad()

        data_1 = data_1.to(device)
        data_2 = data_2.to(device)
        output_1 = model.forward_cl(data_1)
        output_2 = model.forward_cl(data_2)
        loss = criterion(output_1, output_2, training_args.temperature)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader_1.dataset)
    print(f"Average Train Loss: {average_loss}")
    return average_loss


if __name__ == "__main__":
     # task begin...

    # define argments first
    parser = HfArgumentParser((ModelArguments, DataArguments, CLTrainArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # set device
    global device
    if training_args.device == 'gpu':
        device = set_device()
    else:
        device = 'cpu'

    # mkdir for each run
    mkdir(f'./../runs/{training_args.task_name}')

    # load data
    print('*'*20+f'load {data_args.dataset_name}...')
    dataset = MotifDataset(dataset_name=data_args.dataset_name,)
    print('Done!')

    # load model
    if model_args.model_name == 'GIN':
        encoder = GINEncoder
    else:
        encoder = GCNEncoder
    model_args.input_channels = dataset[0].x.size(1)
    # train pipeline
    pipeline(dataset, encoder, data_args, model_args, training_args, )   
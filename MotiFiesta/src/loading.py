import logging
import math
import os

import torch
import torch.distributed as dist
from torch.utils.data import (
    random_split, 
    DataLoader, 
    distributed, 
    RandomSampler, 
    SequentialSampler
)
from torch_geometric.data import Batch

import context
from MotiFiesta.utils.synthetic import SyntheticMotifs
from MotiFiesta.utils.real_world import RealWorldDataset


def get_loader(name=None,
               root="./data",
               batch_size=2,
               num_workers=4,
               local_rank=-1,
               **kwargs
               ):
    print("NAME ",  name)
    print("Num workers ", num_workers)
    if not name.startswith('synth'):
        # if name == 'IMDB-BINARY':
        if name in ['IMDB-BINARY', "REDDIT-BINARY","COLLAB"]:
            dataset = RealWorldDataset(name=name,root=root,)
        else:
            dataset = RealWorldDataset(name=name,root=root)
    else:
        if not root:
            root = f'./data/{name}'
        dataset = SyntheticMotifs(root=root, name=name, **kwargs)

    num_features = dataset[0]['pos'].x.shape[1]

    lengths = [math.floor(len(dataset) * .9), math.ceil(len(dataset) * .1)]
    # lengths = [0.9, 0.1]
    train_data, test_data = random_split(dataset, lengths, generator=torch.Generator())

    # TODO divide data into pos and neg before feed into dataloader
    # loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if local_rank > -1:
        train_sampler = distributed.DistributedSampler(train_data)
        test_sampler = distributed.DistributedSampler(test_data)
        world_size = dist.get_world_size()
        batch_size = batch_size // world_size
    else:
        train_sampler = RandomSampler(train_data)
        test_sampler = SequentialSampler(test_data)
    
    loader_train = DataLoader(
        train_data, 
        collate_fn=collate_fn,
        sampler=train_sampler,
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    loader_test = DataLoader(
        test_data, 
        collate_fn=collate_fn,
        sampler=test_sampler,
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # return {'dataset_whole': dataset, 'loader_whole': loader, 'loader_train': loader_train, 'loader_test': loader_test}
    return loader_train, loader_test, num_features


def get_loader_motif_pred(name=None,
               root="./data",
               batch_size=2,
               num_workers=4,
               local_rank=-1,
               **kwargs
               ):
    print("NAME ",  name)
    print("Num workers ", num_workers)
    if not name.startswith('synth'):
        # if name == 'IMDB-BINARY':
        if name in ['IMDB-BINARY', "REDDIT-BINARY","COLLAB"]:
            dataset = RealWorldDataset(name=name,root=root,)
        else:
            dataset = RealWorldDataset(name=name,root=root)
    else:
        if not root:
            root = f'./data/{name}'
        dataset = SyntheticMotifs(root=root, name=name, **kwargs)

    return dataset


def collate_fn(batch):
    pos_list = []
    neg_list = []
    for b in batch:
        pos_list.append(b['pos'])
        neg_list.append(b['neg'])
    return Batch.from_data_list(pos_list), Batch.from_data_list(neg_list)

if __name__ == "__main__":
    import argparse
    import time
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-d", "--dataset", type=str, default='bbbp')
    args_parser.add_argument("-b", "--batch-size", type=int, default=16)
    args_parser.add_argument("-n", "--num-workers", type=int, default=0)
    args = args_parser.parse_args()
    data = get_loader(
                #   root=hparams['train']['dataset'],
                    name=args.dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                #   attributed=hparams['train']['attributed']
                    )
    print(args)
    train_loader = data[0]
    epoch = 0
    tick = time.time()
    for batch_idx, batch in enumerate(train_loader):
        print(batch[0].x)
        print(batch[0].x.shape)
        print(type(batch[0].x))
        
        exit()
        tok = time.time()
        print(f"Epoch {epoch+1}, Train, Batch {batch_idx} loaded with ({tok-tick})s.")
        tick = tok

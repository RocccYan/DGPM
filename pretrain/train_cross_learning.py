# -*- coding: utf-8 -*-
""""""
import logging
import os
from tqdm import tqdm

import numpy as np
from torch_geometric.data import DataLoader
import torch

from build_dataset import NodeMotifDataset
from configs import DataArguments, ModelArguments, TrainArguments, HfArgumentParser
from utils import (mkdir, load_saved_model, set_device, save_as_json, set_workspace, spotlight2label)
from models import matching_loss, cat_matching_loss
from MotiFiesta.src.decode import HashDecoder


def train_by_similarity(train_loader, model_node, model_motif, optimizer, loss_fn, train_args):
    num_epochs = train_args.num_epochs
    device = train_args.device
    # roc, extract spotlight and embs dict with specific layers
    num_layers = len(list(model_motif.modules())[1]) - 1

    model_node.train()
    model_motif.train()
    epoch_iter = tqdm(range(num_epochs))
    epoch_losses = []
    for epoch in epoch_iter:
        loss_list = []
        for batch in train_loader:
            batch = batch.to(device)
            # inference of model_node
            emb_node = model_node.embed(batch.x, batch.edge_index)
            emb_motif, _, _, _, merge_info, _ = model_motif(batch.x, batch.edge_index, batch.batch)
            emb_motif = emb_motif[-1]
            # labels TODO: extract spotlight by #layersInMotifLearning
            spotlights = merge_info['spotlights'][num_layers]
            labels = spotlight2label(spotlights, mode='sim').to(device)
            loss = loss_fn(emb_node, emb_motif, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            logging.info(f"batch loss: {loss}")
        # TODO
        # if scheduler is not None:
        #     scheduler.step()
        epoch_loss = np.mean(loss_list)
        epoch_losses.append(epoch_loss)
        
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
        logging.info(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
    save_as_json(epoch_losses, f"./models/{train_args.run_name}/loss.json")
    # save model
    return model_node.cpu(), model_motif.cpu()


def train_by_concat(train_loader, model_node, model_motif, optimizer, loss_fn, train_args):
    num_epochs = train_args.num_epochs
    device = train_args.device
    # roc, extract spotlight and embs dict with specific layers
    num_layers = len(list(model_motif.modules())[1]) - 1

    model_node.train()
    model_motif.train()
    epoch_iter = tqdm(range(num_epochs))
    epoch_losses = []
    for epoch in epoch_iter:
        loss_list = []
        for batch in train_loader:
            batch = batch.to(device)
            # inference of model_node
            emb_node = model_node.embed(batch.x, batch.edge_index)
            emb_motif, _, _, _, merge_info, _ = model_motif(batch.x, batch.edge_index, batch.batch)
            emb_motif = emb_motif[-1]
            # labels TODO: extract spotlight by #layersInMotifLearning
            spotlights = merge_info['spotlights'][num_layers]
            labels = spotlight2label(spotlights, mode='cat').to(device)
            # roc, normalization
            emb_node = F.normalize(emb_node,dim=0)
            emb_motif = F.normalize(emb_motif,dim=0)
            loss = loss_fn(emb_node, emb_motif, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            logging.info(f"batch loss: {loss}")
        # TODO
        # if scheduler is not None:
        #     scheduler.step()
        epoch_loss = np.mean(loss_list)
        epoch_losses.append(epoch_loss)
        
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
        logging.info(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
    
        if (epoch+1) % 10 == 0:
            torch.save(model_node.state_dict(), f"./models/{model_args.model_name_node}/model_{train_args.train_mode}_{epoch+1}.pth")
            torch.save(model_motif.state_dict(), f"./models/{model_args.model_name_motif}/model_{train_args.train_mode}_{epoch+1}.pth")
    save_as_json(epoch_losses, f"./models/{train_args.run_name}/loss.json")
    # save model
    return model_node.cpu(), model_motif.cpu()


def main(data_args, model_args, train_args):
    # load data
    dataset = NodeMotifDataset(data_args.dataset_name,)
    dataloader = DataLoader(
        dataset, 
        batch_size=train_args.batch_size,
        num_workers=train_args.num_workers,
        shuffle=True,
    )

    # load pretrained model
    ## graphMAE model for node
    model_node = load_saved_model(
        model_name=model_args.model_name_node, 
        model_type='node',).to(train_args.device)
    ## MotiFiesta model for motif
    model_motif = load_saved_model(
        model_name=model_args.model_name_motif, 
        model_type='motif',).to(train_args.device)
    # optimizer and loss_fn
    optimizer = torch.optim.Adam([
        {'params': model_node.parameters(), 'lr': 0.001,},
        {'params': model_motif.parameters(), 'lr': 0.005,},
        ],)

    # run
    mkdir(f"./models/{train_args.run_name}")
    if train_args.train_mode == 'sim':
        loss_fn = matching_loss
        train_fn = train_by_similarity
        # model_node, model_motif = train_by_similarity(dataloader, model_node, model_motif, optimizer, loss_fn, train_args)
    else:
        loss_fn = cat_matching_loss
        train_fn = train_by_concat
    print("Train Mode:", train_args.train_mode)
    model_node, model_motif = train_fn(dataloader, model_node, model_motif, optimizer, loss_fn, train_args)
    torch.save(model_node.state_dict(), f"./models/{model_args.model_name_node}/{train_args.train_mode}.pth")
    torch.save(model_motif.state_dict(), f"./models/{model_args.model_name_motif}/{train_args.train_mode}.pth")
    logging.info("Training Finished and Models are Saved.")


if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainArguments))
    args = parser.parse_args_into_dataclasses()
    data_args, model_args, train_args = args

    train_args.device = set_device()
    train_args.run_name = f"{train_args.train_mode}+{model_args.model_name_motif}+{model_args.model_name_node}"
    # setup & mkdir 
    set_workspace('/mnt/workspace/graph_pretrain/MotiFiesta')
    
    # set logging 
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(f"./logs/cat/{train_args.run_name}.log"),]
    )

    main(data_args, model_args, train_args)

                                                        
                                                                                   
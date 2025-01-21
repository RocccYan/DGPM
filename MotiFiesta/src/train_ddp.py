import logging
import sys
import time
import math
from collections import defaultdict

from tqdm import tqdm
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torchviz import make_dot
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor
import torch.nn.functional as F

import context
from MotiFiesta.src.model import (rec_loss_fn, freq_loss_fn)
from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.utils.graph_utils import to_graphs
from MotiFiesta.utils.graph_utils import get_subgraphs
from MotiFiesta.utils.pipeline_utils import EarlyStopping, save_as_json

torch.autograd.set_detect_anomaly(True)
    

def set_up_distributed_training(local_rank, model):
    device = torch.device('cuda', local_rank)
    model = DDP(model.to(device),
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,)
    return model,  device
    

def motif_train(model,
                train_loader,
                test_loader,
                model_name='default',
                lr=0.001,
                estimator='kde',
                epochs=5,
                lam=1,
                beta=1,
                max_batches=-1,
                stop_epochs=30,
                volume=False,
                n_neighbors=30,
                hard_embed=False,
                epoch_start=0,
                optimizer=None,
                controller_state=None,
                local_rank=-1,
                ):
    """motif_train.
    """
    start_time = time.time()
    logging.info(f"Train start time:{start_time}")

    if local_rank > -1:
        model, device = set_up_distributed_training(local_rank, model)
    else:
        device = get_device()
        model.to(device)

    # roc, 
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    early_stopper_rec = EarlyStopping(patience=20, min_delta=0)
    early_stopper_mot = EarlyStopping(patience=20, min_delta=0)
    train_loss = {}
    best_loss = 999.9
    for epoch in range(epoch_start, epochs):
        if local_rank > -1:
            train_loader.sampler.set_epoch(epoch)
        model.train()

        num_batches = len(train_loader)
        rec_loss_tot, mot_loss_tot, = [0] * 2
        loss_type = 'mot' if epoch > stop_epochs or early_stopper_rec.early_stop else 'rec'
        # loss_type = 'mot' if epoch > stop_epochs else 'rec'
        # if early_stopper_mot.early_stop:
        if epoch > stop_epochs * 2 or early_stopper_mot.early_stop:
            break
        loss_total = 0
        logging.info(f"Epoch {epoch+1} start. Num batches {num_batches}")
        for batch_idx, (batch_pos, batch_neg) in tqdm(enumerate(train_loader), total=num_batches):
            graphs_pos = to_graphs(batch_pos)
            batch_pos, batch_neg = batch_pos.to(device), batch_neg.to(device)

            optimizer.zero_grad()
            loss = model(batch_pos, batch_neg, graphs_pos,loss_fn=loss_type)  
            print(f"Loss: {loss}")
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        # dist
        if local_rank > -1:
            dist.barrier()

        train_loss.update({f"E={epoch}":loss_total})
        time_elapsed = time.time() - start_time
        print(f"Train Epoch: {epoch+1} [{batch_idx +1}/{num_batches}]"\
              f"({100. * (batch_idx +1) / num_batches :.2f}%) {loss_total}"\
              f" Time: {time_elapsed:.2f}"
              )
        
        # roc, save model with min loss and epoch % 10 == 0
        if local_rank == 0:
            if best_loss > loss_total and loss_total > 0:
                best_loss = loss_total 
                best_model = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({"model_state_dict": best_model},
                f'models/{model_name}/{model_name}_MinLoss.pth')
            if (epoch+1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'controller_state_dict': controller.state_dict()
                }, f'models/{model_name}/{model_name}_{epoch+1}.pth')
        # update earlystopper
        if loss_type == 'rec': 
            early_stopper_rec(loss_total)
        else:
            early_stopper_mot(loss_total)
            
    if local_rank > -1:
        dist.destroy_process_group()

    if local_rank == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            # 'controller_state_dict': controller.state_dict()
        }, f'models/{model_name}/{model_name}.pth')
        save_as_json(train_loss, f'models/{model_name}/train_loss.json')

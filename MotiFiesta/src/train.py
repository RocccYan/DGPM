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
from MotiFiesta.utils.pipeline_utils import save_as_json

torch.autograd.set_detect_anomaly(True)
    

class Controller:
    def __init__(self, since_best_threshold=1):
        self.since_best_threshold = since_best_threshold
        self.modules = ['rec', 'mot']
        self.best_losses = {key: {'best_loss': float('nan'), 'since_best': 0}
                            for key in self.modules}

    def keep_going(self, key):
        """ Returns True if model should keep training, false otherwise."""

        if self.best_losses[key]['since_best'] > self.since_best_threshold:
            return False
        else:
            return True

    def update(self, losses):
        for key, l in losses.items():
            if l < self.best_losses[key]['best_loss']:
                self.best_losses[key]['best_loss'] = l
                self.best_losses[key]['since_best'] = 0
            elif not math.isnan(l):
                self.best_losses[key]['since_best'] += 1
            else:
                pass
        pass

    def state_dict(self):
        return {'since_best_threshold': self.since_best_threshold,
                'modules': self.modules,
                'best_losses': self.best_losses
                }

    def set_state(self, state_dict):
        self.since_best_threshold = state_dict['since_best_threshold']
        self.modules = state_dict['modules']
        self.best_losses = state_dict['best_losses']

def print_gradients(model):
    """
        Set the gradients to the embedding and the attributor networks.
        If True sets requires_grad to true for network parameters.
    """
    for param in model.named_parameters():
        name, p = param
        print(name, p, p.grad, p.requires_grad, p.shape)
    pass


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
                stop_epochs=100,
                volume=False,
                n_neighbors=30,
                hard_embed=False,
                epoch_start=0,
                optimizer=None,
                controller_state=None,
                local_rank=-1,
                ):
    """motif_train.

    :param model: MotiFiesta model
    :param loader: Graph DataLoader
    :param null_loader: optional. loader containing 'null graphs'
    :param model_name: ID to save model under
    :param epochs: number of epochs to train
    :param lambda_rec: loss coefficient for embedding representation loss
    :param lambda_mot: loss coefficient for edge scores
    :param max_batches: if not -1, stop after given number of batches
    """
    start_time = time.time()
    logging.info(f"Train start time:{start_time}")
    # writer = SummaryWriter(f"logs/{model_name}")

    if controller_state is None:
        controller = Controller(since_best_threshold=stop_epochs,)
    else:
        controller = Controller()
        controller.set_state(controller_state)

    device = get_device()
    model.to(device)

    # roc, optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    mot_loss, rec_loss = [torch.tensor(float('nan'))] * 2
    done_training = False
    train_loss = {}
    test_loss = {}
    # roc
    best_loss = 999.9
    for epoch in range(epoch_start, epochs):
        if local_rank > -1:
            train_loader.sampler.set_epoch(epoch)
        if done_training:
            print("DONE TRAINING")
            break
        model.train()
        # only keep parameters that require grad
        num_batches = len(train_loader)
        rec_loss_tot, mot_loss_tot, = [0] * 2
        logging.info(f"Epoch {epoch+1} start. Num batches {num_batches}")
        
        for batch_idx, (batch_pos, batch_neg) in tqdm(enumerate(train_loader), total=num_batches):
            if batch_idx >= max_batches and max_batches > 0:
                break
            # logging.info(f"Epoch {epoch+1}, Train, Batch {batch_idx} start.")
            optimizer.zero_grad()

            graphs_pos = to_graphs(batch_pos)
            batch_pos = batch_pos.to(device)

            x_pos, edge_index_pos, edge_attr_pos = batch_pos.x, batch_pos.edge_index, batch_pos.edge_attr
            xx_pos, pp_pos, ee_pos,_, merge_info_pos, internals_pos = model(x_pos,
                                                                            edge_index_pos,
                                                                            batch_pos.batch,
                                                                            edge_attr=edge_attr_pos)
            loss = 0
            backward = False
            warmup_done = False
            # TODO, can isomophic take edge informations.
            if controller.keep_going('rec') and not hard_embed:
                rec_start = time.time()
                # logging.info(f"Epoch {epoch+1}, Train, Batch {batch_idx}. rec loss start.")
                rec_loss = rec_loss_fn(xx_pos,
                                    ee_pos,
                                    merge_info_pos['spotlights'],
                                    graphs_pos,
                                    batch_pos.batch,
                                    x_pos,
                                    internals_pos,
                                    draw=False)
                rec_loss_tot += rec_loss.item()
                rec_time = time.time() - rec_start
                # logging.info(f"Epoch {epoch+1}, Train, Batch {batch_idx}. rec loss {rec_loss.item()} done with {rec_time}s.")
                backward = True
                loss += rec_loss
            else:
                warmup_done = True

            if controller.keep_going('mot') and warmup_done:
                mot_start = time.time()
                # logging.info(f"Epoch {epoch+1}, Train, Batch {batch_idx}. mot loss start.")
                batch_neg = batch_neg.to(device)
                x_neg, edge_index_neg, edge_attr_neg = batch_neg.x, batch_neg.edge_index, batch_neg.edge_attr
                xx_neg, pp_neg, ee_neg,_, merge_info_neg, internals_neg = model(x_neg,
                                                                            edge_index_neg,
                                                                            batch_neg.batch,
                                                                            edge_attr=edge_attr_neg)
                # logging.info(f"Epoch {epoch+1}, Train, Batch {batch_idx}. forward neg done.")

                mot_loss = freq_loss_fn(internals_pos,
                                    internals_neg,
                                    pp_pos,
                                    steps=model.steps,
                                    estimator=estimator,
                                    volume=volume,
                                    k=n_neighbors,
                                    lam=lam,
                                    beta=beta)
                backward = True
                loss += mot_loss
                mot_loss_tot += mot_loss.item()
                mot_time = time.time() - mot_start
                # logging.info(f"Epoch {epoch+1}, Train, Batch {batch_idx}. mot loss {mot_loss.item()} done with {mot_time}s.")
            if backward:
                if loss.isnan():
                    loss = torch.tensor(1e-6,device=device,requires_grad=True)
                    # logging.info('default mot loss used.')
                loss.backward()
                optimizer.step()
            else:
                done_training = True

        N = max_batches if max_batches > 0 else len(train_loader)

        losses = {'rec': rec_loss_tot / N,
                  'mot': mot_loss_tot / N,}
        logging.info(f"Epoch {epoch+1} Train Loss: {losses}")
        train_loss.update({f'E={epoch+1}':losses})

        ## END OF BATCHES ##
        ## Test ##
        rec_loss_tot, mot_loss_tot, = [0] * 2

        for batch_idx, (batch_pos, batch_neg) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if batch_idx >= max_batches and max_batches > 0:
                break
            logging.info(f"Epoch {epoch+1}, Test, Batch {batch_idx} start.")
            model.eval()

            graphs_pos = to_graphs(batch_pos)
            batch_pos = batch_pos.to(device)
            x_pos, edge_index_pos = batch_pos.x, batch_pos.edge_index
            
            with torch.no_grad():
                x_pos, edge_index_pos, edge_attr_pos = batch_pos.x, batch_pos.edge_index, batch_pos.edge_attr
                xx_pos, pp_pos, ee_pos,_, merge_info_pos, internals_pos = model(x_pos,
                                                                                edge_index_pos,
                                                                                batch_pos.batch,
                                                                                edge_attr=edge_attr_pos)
            warmup_done = False
            if controller.keep_going('rec'):
                rec_loss = rec_loss_fn(xx_pos,
                                    ee_pos,
                                    merge_info_pos['spotlights'],
                                    graphs_pos,
                                    batch_pos.batch,
                                    x_pos,
                                    internals_pos
                                    )
            else:
                warmup_done = True

            if warmup_done:
                batch_neg = batch_neg.to(device)
                x_neg, edge_index_neg = batch_neg.x, batch_neg.edge_index
                with torch.no_grad():
                    x_neg, edge_index_neg, edge_attr_neg = batch_neg.x, batch_neg.edge_index, batch_neg.edge_attr
                    xx_neg, pp_neg, ee_neg,_, merge_info_neg, internals_neg = model(x_neg,
                                                                                edge_index_neg,
                                                                                batch_neg.batch,
                                                                                edge_attr=edge_attr_neg)

                mot_loss = freq_loss_fn(internals_pos,
                                        internals_neg,
                                        pp_pos,
                                        steps=model.steps,
                                        estimator=estimator,
                                        volume=volume,
                                        k=n_neighbors,
                                        lam=lam,
                                        beta=beta
                                        )
            rec_loss_tot += rec_loss.item()
            mot_loss_tot += mot_loss.item()
            # logging.info(f"Epoch {epoch+1}, Test, Batch {batch_idx} done.")

        N = max_batches if max_batches > 0  else len(test_loader)
        test_losses = {'rec': rec_loss_tot / N,
                    'mot': mot_loss_tot / N,}
        test_loss.update({f'E={epoch+1}':test_losses})
        controller.update(test_losses)

        # roc, save model with min loss 
        if best_loss > test_losses['rec']:
            best_loss = test_losses['rec'] 
            best_model = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_model},
            f'models/{model_name}/{model_name}_MinLoss.pth')
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                # 'model_state_dict': model.state_dict(),
                'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'controller_state_dict': controller.state_dict()
            }, f'models/{model_name}/{model_name}_{epoch+1}.pth')

        loss_str = ' '.join([f'{k} train: {v:2f}' for k,v in losses.items()])
        test_loss_str = ' '.join([f'{k} test: {v:2f}' for k,v in test_losses.items()])
        time_elapsed = time.time() - start_time
        print(f"Train Epoch: {epoch+1} [{batch_idx +1}/{num_batches}]"\
              f"({100. * (batch_idx +1) / num_batches :.2f}%) {loss_str}"\
              f" {test_loss_str}"\
              f" Time: {time_elapsed:.2f}"
              )
    save_as_json(train_loss,f'models/{model_name}/train_loss.json')
    save_as_json(test_loss,f'models/{model_name}/test_loss.json')

    model.cpu()
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'controller_state_dict': controller.state_dict()
    }, f'models/{model_name}/{model_name}.pth')

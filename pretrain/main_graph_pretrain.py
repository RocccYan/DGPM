import logging
from tqdm import tqdm
import numpy as np
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_graph_classification_dataset
from graphmae.models import build_model
import context
from MotiFiesta.utils.pipeline_utils import set_workspace, mkdir, save_as_json


def pretrain(model, pooler, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob=True, logger=None):
    train_loader, eval_loader = dataloaders

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batch in train_loader:
            batch_g = batch
            batch_g = batch_g.to(device)

            feat = batch_g.x
            model.train()
            loss, loss_dict = model(feat, batch_g.edge_index)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.note(loss_dict, step=epoch)
        if scheduler is not None:
            scheduler.step()
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

    return model

            


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    pooler = args.pooling
    deg4feat = args.deg4feat
    batch_size = args.batch_size

    graphs, (num_features, num_classes) = load_graph_classification_dataset(dataset_name, deg4feat=deg4feat)
    args.num_features = num_features

    train_idx = torch.arange(len(graphs))
    
    train_loader = DataLoader(graphs, batch_size=batch_size, pin_memory=True)
    eval_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)

    if use_scheduler:
        logging.info("Use schedular")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None
        
    if not load_model:
        model = pretrain(model, pooler, (train_loader, eval_loader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob,)
        model = model.cpu()

    if save_model:
        # logging.info("Saveing Model ...")
        print("Saveing Model ...")
        run_name = f"{dataset_name}_graphmae"
        mkdir(f"./models/{run_name}")
        save_as_json(vars(args),f"./models/{run_name}/hparams.json")
        torch.save(model.state_dict(), f"./models/{run_name}/model.pth")
    
    # ${dataset}_D=${D}_B=${B}_E=${E}_W=${W}

if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    set_workspace("/mnt/workspace/graph_pretrain/MotiFiesta/")
    main(args)
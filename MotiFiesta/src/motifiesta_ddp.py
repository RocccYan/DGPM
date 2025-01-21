import argparse
import logging
import pickle
import sys
import warnings

import numpy as np
import torch
import torch.distributed as dist

import sys
sys.path.append("/mnt/workspace/graph_pretrain")
from MotiFiesta.src.model_ddp import MotiFiesta
from MotiFiesta.src.loading import get_loader
from MotiFiesta.src.train_ddp import motif_train
from MotiFiesta.utils.learning_utils import dump_model_hparams
from MotiFiesta.utils.learning_utils import load_model
from MotiFiesta.utils.learning_utils import make_dirs
from MotiFiesta.utils.pipeline_utils import set_workspace, set_seed
# from MotiFiesta.utils.learning_utils import load_data

set_workspace("MotiFiesta")
set_seed(42)
warnings.simplefilter("ignore", UserWarning)


def init_distributed_training(local_rank):
    # dist.init_process_group(backend='nccl')
    world_size = torch.cuda.device_count()
    print(f"world_size:{world_size}")
    dist.init_process_group('nccl', rank=local_rank, world_size=world_size)
    # dist.barrier()
    if dist.get_rank() == 0:
        logging.info(f'Global Number of Processing: {world_size}')


parser = argparse.ArgumentParser()
parser.add_argument("--name", "-n", type=str, default="default")
parser.add_argument("--dataset", "-da", type=str, default='synthetic')
parser.add_argument("--restart", "-r", action='store_true', default=False,  help='Restart model.')

# training  loop
parser.add_argument("--batch-size", "-b", type=int, default=16)
parser.add_argument("--max-batches", "-m", type=int, default=-1)
parser.add_argument("--epochs", "-e", type=int, default=200)
parser.add_argument("--stop-epochs", "-se", type=int, default=30, help="Number of epochs to train embeddings before doing motifs.")
parser.add_argument("--attributed", default=False, action='store_true', help="Use attributed graphs. If False, use node degree as features.")
parser.add_argument("--num-workers", "-w", type=int, default=6)
parser.add_argument("--local_rank", type=int, default=-1)

# learning
parser.add_argument("--lr", '-lr',  type=float, default=0.001)
parser.add_argument("--lam",  type=float, default=1)
parser.add_argument("--beta", type=float, default=1)
parser.add_argument("--estimator", "-es", type=str, default='knn', help="Which density estimator to use for the motif agg. step")
parser.add_argument("--n-neighbors", "-nn", type=int, default=30, help="Number of neighbors")
parser.add_argument("--volume", "-vv", action='store_true', default=False, help="Use d-sphere volume to normalize density with kNN.")

# model architecture
parser.add_argument("--steps", "-s", type=int, default=5)
parser.add_argument("--dim", "-d", type=int, default=8)
parser.add_argument("--pool-dummy", action="store_true", default=False)
parser.add_argument("--score-method", default="sigmoid", help="Edge soring method: (sigmoid, softmax-neighbors, softmax-all)")
parser.add_argument("--merge-method", default="sum", help="Edge merge method: (sum, cat, set2set)")
parser.add_argument("--hard-embed", action='store_true', help="Whether to use hard embedding using degree histogram. ")

args, _ = parser.parse_known_args()
# roccc run name, in case run name is not in line with real paras.
# run_name = f"{args.dataset}_D={args.dim}_B={args.batch_size}_E={args.epochs}_W={args.num_workers}"

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.FileHandler(f"./logs/{args.name}.log"),]
)
logging.info(args)

make_dirs(args.name)

hparams = {'model':{
                    'dim': args.dim,
                    'steps': args.steps,
                    'pool_dummy': args.pool_dummy,
                    'edge_score_method': args.score_method,
                    'merge_method': args.merge_method,
                    'hard_embed': args.hard_embed,
                        },
            'train': {
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'max_batches': args.max_batches,
                    'dataset': args.dataset,
                    'lambda': args.lam,
                    'beta': args.beta,
                    'stop_epochs': args.stop_epochs,
                    'estimator': args.estimator,
                    'k': args.n_neighbors,
                    'volume': args.volume,
                    'attributed': args.attributed,
                    'num_workers': args.num_workers,
                    }
            }

# TODO whether distributed 
local_rank = args.local_rank
# print(f"local rank: {local_rank}")
# logging.info(f"local rank: {local_rank}")
if local_rank > -1:
    init_distributed_training(local_rank)
    logging.info(f"local rank: {local_rank} initialized! ")

print(">>> loading data")
logging.info(">>> loading data")
data = get_loader(
                #   root=hparams['train']['dataset'],
                    name=hparams['train']['dataset'],
                    batch_size=hparams['train']['batch_size'],
                    num_workers=hparams['train']['num_workers'],
                #   attributed=hparams['train']['attributed']
                    local_rank=local_rank,
                    )

loader_train, loader_test, num_features = data

logging.info(f"dataset {args.dataset} loaded.")
hparams['model']['n_features'] = num_features
print(num_features)
dump_model_hparams(args.name, hparams)

print(">>> building model")
logging.info(">>> building model")
if args.restart:
    print(f"Restarting training with ID: {args.name}")
    model_dict = load_model(args.name, ddp_flag=True)
    model = model_dict['model']
    # exit()
    epoch_start = model_dict['epoch']
    optimizer = model_dict['optimizer']
    controller_state = model_dict['controller_state_dict']
else:
    model = MotiFiesta(**hparams['model'])
    epoch_start, optimizer, controller_state = 0, None, None

print(model)
logging.info(model)
print(">>> training...")
logging.info(">>> training...")
motif_train(model,
            train_loader=loader_train,  
            test_loader=loader_test,
            model_name=args.name,
            lr=args.lr,
            epochs=args.epochs,
            max_batches=args.max_batches,
            stop_epochs=args.stop_epochs,
            estimator=args.estimator,
            volume=args.volume,
            n_neighbors=args.n_neighbors,
            hard_embed=args.hard_embed,
            beta=args.beta,
            lam=args.lam,
            epoch_start=epoch_start,
            optimizer=optimizer,
            controller_state=controller_state,
            local_rank=local_rank,
            )


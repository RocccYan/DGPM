"""
hparams need to be finetuned includes: 
    1. model: model_type, hidden_channels, layers, readout,
    2. training: batch_size, lr, temperature, num_epoches, 
"""

from multiprocessing import Pool
import os


DEVICE = 'gpu'
DATASETS = [
    # "COX2",
    # "PROTEINS",
    # "IMDB-BINARY",
    "MUTAG",
]

task_datasets = DATASETS
# cl_modes = ['simgrace','gcl']
# models = ['GIN','OGCN','ResGCN']
models = ['GIN','OGCN',]
num_layers = 5
hidden_channels = 128
# temperatures = [0.01, 0.001]
temperatures = [0.5,]
batch_sizes = [512]
# simgrace lr [0.1, 1.0, 5.0, 10.0]
learning_rates = [0.0001, 0.01, 0.1]

num_train_epochs = 100
print_per_epoch = 20

# Add to pool 
p = Pool(8)
for dataset_name in task_datasets:
    for model_name in models:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                for temp in temperatures:
                    #********************
                    task_name = f'{dataset_name}_{model_name}_B={batch_size}_lr={lr}_T={temp}'
                    print(f'Training *{task_name}* added ...')
                    p.apply_async(os.system,(
                    # os.system(
                        f"python train.py "
                        f"--dataset_name {dataset_name} "
                        f"--model_name {model_name} "
                        f"--hidden_channels {hidden_channels} "
                        f"--num_layers {num_layers} "
                        f"--lr {lr} "
                        f"--temperature {temp} "
                        f"--batch_size {batch_size} "
                        f"--num_epochs {num_train_epochs} "
                        f"--device {DEVICE} "
                        f"--task_name {task_name} "
                        f"--print_per_epoch {print_per_epoch} ",)
                        )

p.close()
p.join()






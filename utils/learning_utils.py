import os
import json

import torch

device_cache = None
def get_device():
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        #device_cache = torch.device("cpu")
    return device_cache

def load_data(run, batch_size=2, background_only=False):
    with open(f'./models/{run}/hparams.json', 'r') as j:
        json_params = json.load(j)
    data = dataset_from_json(json_params)
    return data

def load_model(run, permissive=False, verbose=True, ddp_flag=False):
    """
    Input the name of a run
    :param run:
    :return:
    """
    with open(f'./models/{run}/hparams.json', 'r') as j:
        json_params = json.load(j)

    model = model_from_json(json_params,ddp=ddp_flag)

    try:
        # roc, 
        model_files = sorted([f for f in os.listdir(f'./models/{run}') if f.endswith('.pth')])
        model_dict = torch.load(f'./models/{run}/{model_files[-2]}',map_location='cpu')
        # model_dict = torch.load(f'./models/{run}/{run}.pth',map_location='cpu')
        state_dict = model_dict['model_state_dict']
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        # print(state_dict)
        model.load_state_dict(state_dict)
        print("*"*8+f' ./models/{run}/{model_files[-2]} loaded!'+"*"*8)
        # exit()
        if "optimizer_state_dict" in model_dict:
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        else:
            optimizer = None
        

    except FileNotFoundError:
        if not permissive:
            raise FileNotFoundError('There are no weights for this experiment...')
    return {'model': model,
            'epoch': model_dict.get('epoch', 0),
            'optimizer':optimizer,
            'controller_state_dict': model_dict.get('controller_state_dict', None)
            }

def dump_model_hparams(name, hparams):
    with open(f'./models/{name}/hparams.json', 'w') as j:
        json.dump(hparams, j)
    pass

def model_from_json(params,ddp=False):
    if ddp:
        from MotiFiesta.src.model_ddp import MotiFiesta  
    else:
        from MotiFiesta.src.model import MotiFiesta
    model = MotiFiesta(**params['model'])
    return model

def dataset_from_json(params, background=False):
    from MotiFiesta.src.loading import get_loader
    data = get_loader(root=params['train']['dataset'],\
                                batch_size=params['train']['batch_size']\
                                )
    return data

def make_dirs(run):
    try:
        os.mkdir(f"./models/{run}")
    except FileExistsError:
        pass

def one_hot_to_id(x):
    """ Create column vector with index where one-hot is 1. """
    return torch.nonzero(x, as_tuple=True)[1]

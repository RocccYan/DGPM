"""
Split supervised datasets into 10-flods and preserved trian/test idx into .txt files.
../supervised_data/fold-idx/$dataset_name/train[test]_idx-$fold-idx.txt
"""
import sys
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


def split_dataset(labels,dataset_name,n_splits=10,seed=42):
    # check first
    fold_idx_path = f'../supervised_data/fold-idx/{dataset_name}/'
    if os.path.exists(fold_idx_path) and os.listdir(fold_idx_path):
        return 
    else:
        os.makedirs(fold_idx_path, exist_ok=True)

    idxpath = '../supervised_data/fold-idx/{}/{}_idx-{}.txt'
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx_list = []
    for fold_idx, idx in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # idx_list.append(idx)
        train_idx, test_idx = idx
        np.savetxt(idxpath.format(dataset_name, 'train', fold_idx),train_idx,delimiter=',',fmt='%d')
        np.savetxt(idxpath.format(dataset_name, 'test', fold_idx),test_idx,delimiter=',',fmt='%d')
        


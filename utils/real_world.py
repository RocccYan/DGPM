"""
    Generate graphs containing synthetic motifs.
    The idea:
        1. A motif is a random graph
        2. For each instance, insert motif graph into larger random graph.
        4. Insertion means sampling a random node from parent graph and replacing it with motif.
        5. Links between motif and parent graph are made by sampling random nodes in both subgraphs with same probability
           as the edge probability which generated the graphs.
        6. Repeat for each motif.
"""
import os
import os.path as osp
import random
import itertools

import numpy as np
from numpy.random import normal
from scipy.stats import beta
from scipy.stats import uniform
from scipy.stats import expon
import networkx as nx
from networkx.generators.random_graphs import connected_watts_strogatz_graph
from networkx.generators.random_graphs import powerlaw_cluster_graph
from networkx.generators.random_graphs import extended_barabasi_albert_graph
from networkx.generators.random_graphs import erdos_renyi_graph
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.datasets import TUDataset, ZINC #, BA2MotifDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx, degree
from torch_geometric.utils import to_networkx

import sys
sys.path.append("/mnt/workspace/graph_pretrain")
from MotiFiesta.utils.molecule_dataset import MoleculeDataset
# from molecule_dataset import NCI1Dataset


TUDATASETS = ['MUTAG','EZYMES','PROTEINS','COLLAB','IMDB-BINARY','REDDIT-BINARY','ZINC-full', "NCI1"]

def rewire(g_pyg, n_iter=100):
    """ Apply (u, v), (u', v') --> (u, v'), (v, u') to randomize graph.
    """
    has_features = g_pyg.x is not None
    if has_features:
        g_nx = to_networkx(g_pyg, node_attrs=['x'])
    else:
        g_nx = to_networkx(g_pyg)
    rewired_g = g_nx.copy()
    for n in range(n_iter):
        e1, e2 = random.sample(list(g_nx.edges()), 2)
        rewired_g.remove_edges_from([e1, e2])
        rewired_g.add_edges_from([(e1[0], e2[1]), (e1[1], e2[0])])

    rewired_g.remove_edges_from(list(nx.selfloop_edges(rewired_g)))
    if has_features:
        rewired_pyg = from_networkx(g_nx, group_node_attrs=['x'])
    else:
        rewired_pyg = from_networkx(g_nx)
    return rewired_pyg


class RealWorldDataset(Dataset):
    def __init__(self,
                 name="ENZYMES",
                 root='./data',
                 n_swap=100,
                 transform=None,
                 seed=42,
                 max_degree=400,
                 n_features=None):
        """ Builds the synthetic motif dataset. Motifs are built on the
        fly and stored to disk.

        Args:
        ---
        name (str): path to folder where graphs will be stores.
        n_graphs (int): number of graphs to generate
        n_motifs (int): number of motifs to inject in graphs
        """
        self.seed = seed
        self.n_swap = n_swap
        self.max_degree = max_degree
        
        # if name in ['NCI1']:
        #     self.base_data = NCI1Dataset(root='./data/NCI1')
        if name in ["ZINC"]:
            self.base_data = MoleculeDataset("./data/" + 'zinc_standard_agent', dataset='zinc_standard_agent')
        elif name in ["ZINC_250k"]:
            sample_size = 250000
            self.base_data = MoleculeDataset("./data/" + 'zinc_standard_agent', dataset='zinc_standard_agent')
            dataset_size = len(self.base_data)
            self.base_data = self.base_data[torch.randperm(dataset_size)[:sample_size]]
        elif name in ["ZINC_20k"]:
            self.base_data = MoleculeDataset("./data/" + 'zinc_standard_agent', dataset='zinc_standard_agent')[:20000]
        elif name in ["ZINC_1k"]:
            self.base_data = MoleculeDataset("./data/" + 'zinc_standard_agent', dataset='zinc_standard_agent')[:1000]
        elif name in ['bbbp']:
            self.base_data = MoleculeDataset("./data/" + 'bbbp', dataset='bbbp')
        elif name in ["BA2Motif"]:
            self.base_data = LocalDataset(root=root, name=name)
        else:
            self.base_data = TUDataset(root=root, name=name)

        if self.base_data[0].x is None:
            self.num_features_degree = self.degree2feature()
        # roc, check x dtype
        # print(self.base_data[0].x.dtype != torch.float)
        # print(self.base_data[0].x != None)
        # exit()
  
        super(RealWorldDataset, self).__init__(os.path.join(root,name), transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        # self.toy_data = torch.load(osp.join(self.processed_dir, f'data_0.pt'))

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.base_data))]

    @property
    def num_features(self):
        if self.base_data.num_features:
            return self.base_data.num_features
        else:
            return self.num_features_degree

    def degree2feature(self):
        print("Using degree as node features")
        feature_dim = 0
        degrees = []
        for g in self.base_data:
            feature_dim = max(feature_dim, degree(g.edge_index[0]).max().item())
            degrees.extend(degree(g.edge_index[0]).tolist())
        feature_dim = min(feature_dim, self.max_degree)
        return feature_dim + 1

    def process(self):
        pass

    # def __len__(self):
    def len(self):
        return len(self.processed_file_names)

    # def __getitem__(self, idx):
    def get(self, idx):
        """ Returns dictionary where 'pos' key stores batch with
        graphs that contain the motif, and the 'neg' key has batches
        without the motif.
        """
        if idx > len(self) - 1:
            raise StopIteration
        g = self.base_data[idx]

        # roc, onehot
        if g.x is None:
            # T.OneHotDegree(self.max_degree)(g)
            degrees = degree(g.edge_index[0])
            degrees[degrees > self.max_degree] = self.max_degree
            degrees = torch.Tensor([int(x) for x in degrees.numpy().tolist()])
            feat = F.one_hot(degrees.to(torch.long), num_classes=int(self.num_features)).float()
            g.x = feat
        # roc, 0813, dont change it if it's categorical variables.
        # elif g.x.dtype != torch.float:
        #     g.x = g.x.float()
        g_neg = rewire(g, n_iter=self.n_swap)
        g.motif_id = torch.zeros(g.num_nodes)
        g_neg.motif_id = torch.zeros(g.num_nodes)
        
        return {'pos': g, 'neg': g_neg}


class LocalDataset(InMemoryDataset):

    def __init__(self,
                 root='./data',
                 name=None,
                 ):
        super().__init__(root=os.path.join(root, name),)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return ['data.pt']


if __name__ == "__main__":

    # d = SyntheticMotifs(motif_type='barbell', distort_p=.02, motif_size=8, root="barbell-pair-s")
    # d = LocalDataset(root='./data',name='BA2Motif')
    os.chdir('/mnt/workspace/graph_pretrain/MotiFiesta/')
    d = RealWorldDataset(root='./data',name='bbbp')
    print(d[0]['pos'])
    print(d[0]['pos'].x)
    print(d[0]['pos'].edge_attr)

'''
    
    
'''
"""
    build graph dataset with motif.
    created by roc.
"""
import copy
from collections import defaultdict
import itertools
import os
import os.path as osp

import networkx as nx
import numpy as np
from random import randrange
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx, to_networkx, degree
from tqdm import tqdm

import context
from MotiFiesta.src.loading import get_loader
from MotiFiesta.utils.pipeline_utils import read_json, set_workspace
from MotiFiesta.utils.real_world import RealWorldDataset


class MotifDataset(Dataset):
    r"""Build graph dataset with learned motif, by loading dataset built 
    during motif learning with `get_loader`, and use the learned motif 
    to build a subgraph dataset with motif as labels.
    """
    def __init__(self, 
                 dataset_name,
                 root=None, 
                 transform=None, 
                 ):
        self.dataset_name = dataset_name
        self.base_data = None
        self.motifs = None
        self.size = 0
        if not root:
            root = './../data'
        super().__init__(osp.join(root,f"{dataset_name}_MOTIF"), transform)

    @property
    def raw_file_names(self):
        "raw file has been loaded in __init__."
        pass

    @property
    def processed_file_names(self):
        """this can be tricky, what if you dont know the specific number of data after processing,
        how could you know or predefine all the processed file names.
        """
        # TODO: temp condition
        return [f'data_{i}.pt' for i in range(1)]
    
    def process(self):
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]
        self.base_data = get_loader(name=self.dataset_name).get('dataset_whole',[])
        self.motifs = load_learned_motifs(self.dataset_name)
        datalist_with_motif = list(map(lambda graph: 
                build_subgraph_with_node_motif(
                    self.base_data[int(graph.split("_")[1])]['pos'], 
                    self.motifs.get(graph),
                    int(graph.split("_")[1])),
                self.motifs.keys()))
        datalist_with_motif = sum(datalist_with_motif,[])
        self.size = len(datalist_with_motif)
        for idx, graph_with_motif in enumerate(datalist_with_motif):
            torch.save(graph_with_motif, osp.join(self.processed_dir, f"data_{idx}.pt"))
        
    def len(self):
        return len(list(filter(lambda x: 'data' in x, os.listdir(self.processed_dir))))

    def get(self, idx):
        if idx > len(self) - 1:
            raise StopIteration
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


def load_learned_motifs(dataset_name):
    r"""motif files are stored at 
    MotiFiesta/results/motif_assignment_{dataset_name}."""
    motif_path = f"./results/motif_assignments_{dataset_name}.json"
    motifs = read_json(motif_path)
    return motifs


def assign_graph_motif(graph, motif):
    r"""assign motif as a graph label to the graph, and also assign graph
    with single node as None.
    """
    if graph.x.size(0) == 1:
        return None
    graph.motif = torch.tensor(motif,dtype=int)
    return graph


def split_graph_by_motif(graph_nx,nodes_motif,target_motif):
    target_motif_nodes = list(map(lambda x:x[0] if x[1]==target_motif else -1,
        zip(graph_nx.nodes(),nodes_motif)))
    m_subgraphs_nodes = list(nx.weakly_connected_components(
        graph_nx.subgraph(target_motif_nodes)))
    tmp_graph_list = list(map(lambda x: from_networkx(
        graph_nx.subgraph(x)), m_subgraphs_nodes))
    tmp_graph_list = list(map(lambda x: assign_graph_motif(
        x, target_motif), tmp_graph_list))
    tmp_graph_list = list(filter(lambda x: x, tmp_graph_list))
    return tmp_graph_list


def build_subgraph_with_node_motif(graph, nodes_motif, graph_idx=0):
    r"""disentangle a graph data into a series of motif-based subgraphs.
    Args:
        graph(torch_geometric.data):
        nodes_motif(list of integers): list stores the motif ID of each node.
    Return:
        subgraphs (list of torch_geometric.data): graphs attributed by motif ID.
    """
    graph_nx = to_networkx(graph, node_attrs=['x'], to_undirected=False)
    motifs = set(nodes_motif)
    motif_subgraph_list = sum(list(map(lambda x: split_graph_by_motif(
        graph_nx, nodes_motif, x), motifs)),[])
    # TODO: assign original graph index to subgraphs.
    def assign_subgraph_with_origin_idx(subgraph, origin_idx):
        subgraph.origin_idx = torch.tensor(graph_idx,dtype=int)
        return subgraph
        
    motif_subgraph_list = list(map(lambda x:
        assign_subgraph_with_origin_idx(x, graph_idx), motif_subgraph_list))
    return motif_subgraph_list


def shuffle_with_fixed_motif_order(dataset):
    r"""return a shuffled dataset or datalist, keep the sequence of motifs 
    of graphs as the original one.
    """
    motif_sequence = list(map(lambda x: int(x.motif.numpy()), dataset))
    motif_data_dict = defaultdict(list)
    for data in tqdm(dataset):
        motif_data_dict[int(data.motif.numpy())].append(data)

    dataset_shuffled = []
    for motif in tqdm(motif_sequence):
        dataset_shuffled.append(
            motif_data_dict[motif].pop(randrange(len(motif_data_dict[motif]))))
    return dataset_shuffled



class NodeMotifDataset(Dataset):
    r"""Build graph dataset in which nodes are assigned with affiliated motif ID.
    """
    def __init__(self, 
                 name,
                 motifs=None,
                 root='./data', 
                 max_degree=400,
                 transform=None, 
        ):
        self.name = name
        self.max_degree = max_degree
        # TODO base data
        self.base_data = RealWorldDataset(name=name,root=root).base_data
        # roc
        # self.motifs = motifs if motifs else load_learned_motifs(self.name)
        if self.base_data[0].x is None:
            self.num_features_degree = self.degree2feature()
        super().__init__(osp.join(root,f"{name}_NODE_MOTIF"), transform)

    @property
    def processed_file_names(self):
        return [f'data.pt']
    
    def process(self):
        pass
        
    def len(self):
        return len(self.base_data)

    @property
    def num_features(self):
        if self.base_data.num_features:
            return self.base_data.num_features
        else:
            return self.num_features_degree

    def degree2feature(self):
        print(f"**{self.name}** using **degree** as node features")
        feature_dim = 0
        degrees = []
        for g in self.base_data:
            feature_dim = max(feature_dim, degree(g.edge_index[0]).max().item())
            degrees.extend(degree(g.edge_index[0]).tolist())
        feature_dim = min(feature_dim, self.max_degree)
        return feature_dim + 1

    def get(self, idx):
        g = self.base_data[idx]
        # roc, 20230511, motif should not be predefined.
        # g.motifs = torch.tensor(self.motifs[f"graph_{idx}"], dtype=torch.long)
        if g.x is None:
            # T.OneHotDegree(self.max_degree)(g)
            degrees = degree(g.edge_index[0])
            degrees[degrees > self.max_degree] = self.max_degree
            degrees = torch.Tensor([int(x) for x in degrees.numpy().tolist()])
            feat = F.one_hot(degrees.to(torch.long), num_classes=int(self.num_features)).float()
            g.x = feat
        return g


if __name__ == "__main__":

    # for dataset in ["COX2","PROTEINS","IMDB-BINARY",]:
    #     dataset_motif = MotifDataset(root='./../data',dataset_name=dataset)
    #     print(len(dataset_motif))
    #     print(dataset_motif[0])
    set_workspace()
    for dataset in ["COX2","PROTEINS","IMDB-BINARY",]:
        dataset_motif = NodeMotifDataset(name=dataset,root='./data',)
        print(len(dataset_motif))
        print(dataset_motif[0])
        print(dataset_motif[0].motifs)
        
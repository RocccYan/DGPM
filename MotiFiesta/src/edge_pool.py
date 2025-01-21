import random
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Set2Set, MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, degree, softmax


# edge pooling for chem
class EdgePooling(torch.nn.Module):
    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index",
         "cluster",
         "batch",
         "new_edge_score",
         "old_edge_score"])

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_score_method='sigmoid',
                 dropout=0,
                 merge_method='sum',
                 add_to_edge_score=0.0,
                 conv_first=False,
                 ):
        super(EdgePooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_first = conv_first

        if edge_score_method == 'softmax':
            edge_score_method = self.compute_edge_score_softmax
        elif edge_score_method == 'sigmoid':
            edge_score_method = self.compute_edge_score_sigmoid
        else:
            edge_score_method = self.compute_edge_score_softmax_full

        if merge_method == 'cat':
            self.edge_merge = self.merge_edge_cat
        else:
            self.edge_merge = self.merge_edge_sum

        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout
        self.merge_method = merge_method
        
        dim = 2 if merge_method == 'cat' else 1
        # compute merged embeddings
        self.transform = torch.nn.Linear(dim * in_channels, out_channels)
        # scoring layer
        self.score_net = torch.nn.Linear(out_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.score_net.reset_parameters()
        self.transform.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes, batch):
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_softmax_full(raw_edge_score, edge_index, num_nodes, batch):
        e_batch = batch[edge_index[0]]
        return softmax(raw_edge_score, e_batch)

    @staticmethod
    def compute_edge_score_dummy(raw_edge_score, edge_index, num_nodes, batch):
        return torch.tensor([.5] * edge_index.shape[1], dtype=torch.float)

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index, num_nodes, batch):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index, num_nodes, batch):
        return torch.sigmoid(raw_edge_score)
    @staticmethod
    def merge_edge_cat(x, edge_index):
        return torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)

    @staticmethod
    def merge_edge_sum(x, edge_index, edge_emb=None):
        # X = x[torch.flatten(edge_index.T)]
        # batch = torch.arange(0, len(edge_index[0])).repeat_interleave(2)
        # batch = batch.to(X.device)
        x_merged = x[edge_index[0]] + x[edge_index[1]]
        if edge_emb != None:
            return x_merged + edge_emb
        else:
            return x_merged

    def forward(self, x, edge_index, batch, edge_emb=None, hard_embed=False, dummy=False):
        # x = x.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # add edge emb, TODO: edge_attr just need encode once
        # if edge_emb is None:
        #     edge_emb = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        
        x_merged = self.edge_merge(x, edge_index, edge_emb)
        x_merged = self.transform(x_merged)

        # compute features for each node with itself in case node is not pooled
        e_ind_self = torch.tensor([list(range(len(x))), list(range(len(x)))])
        x_merged_self = self.edge_merge(x, e_ind_self)
        x_merged_self = self.transform(x_merged_self)

        # compute scores for each edge
        e = self.score_net(x_merged).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0), batch)

        if dummy:
            e = torch.full(e.shape, .5, dtype=torch.float32)

        x_new, edge_index, batch, unpool_info, new_edge_emb = self.__merge_edges__(
            x, edge_index, batch, e, x_merged, x_merged_self, edge_emb=edge_emb)

        return {'new_graph': {'x_new': x_new, 'e_ind_new': edge_index, 'batch_new': batch, 'unpool': unpool_info, 'edge_emb_new': new_edge_emb},
                'internals': {'x_merged': x_merged, 'x_merged_self': x_merged_self, 'edge_scores': e}
                }

    def __merge_edges__(self, x, edge_index, batch, edge_score, x_merged, x_merged_self, edge_emb=None):
        # roc, update edge_attr
        nodes_remaining = set(range(x.size(0)))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        edge_argsort = torch.argsort(edge_score, descending=True)

        i = 0
        new_edge_indices = []
        emb_cat = []
        merge_count = 0

        edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            r = random.random()
            if r > edge_score[edge_idx]:
                continue

            source = edge_index_cpu[0, edge_idx].item()
            if source not in nodes_remaining:
                continue
            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            merge_count += 1
            emb_cat.append(x_merged[edge_idx])
            new_edge_indices.append(edge_idx)

            cluster[source] = i
            nodes_remaining.remove(source)

            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)

            i += 1

        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            emb_cat.append(x_merged_self[node_idx])
            i += 1

        # roc,
        # new_x = torch.zeros((len(emb_cat), len(emb_cat[0])), dtype=torch.float)
        # new_x = new_x.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # for ind, emb in enumerate(emb_cat):
        #     new_x[ind] = emb
        new_x = torch.stack(emb_cat).to(x.device)

        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])

        cluster = cluster.to(x.device)
        N = new_x.size(0)

        new_edge_index, new_edge_emb = coalesce(cluster[edge_index], edge_emb, N, N)    
        new_edge_index, new_edge_emb = remove_self_loops(new_edge_index, new_edge_emb)  

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch,
                                              new_edge_score=new_edge_score,
                                              old_edge_score=edge_score)

        return new_x, new_edge_index, new_batch, unpool_info, new_edge_emb

    def unpool(self, x, unpool_info):
        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.in_channels)


if __name__ == "__main__":
    ep = EdgePooling(3, 2, edge_score_method='sigmoid')
    edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[1, 0, 1], [0, 2, -1], [-1, 0, 1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    # edge_emb = torch.rand(4,3)
    edge_emb = None
    o = ep(data.x, data.edge_index, torch.tensor([0] * len(x), dtype=torch.long), edge_emb=edge_emb,)
    print(o)
    pass

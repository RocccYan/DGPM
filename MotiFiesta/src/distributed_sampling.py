import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

import context
from MotiFiesta.src.model import (rec_loss_fn, freq_loss_fn)
from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.utils.graph_utils import to_graphs
from MotiFiesta.utils.graph_utils import get_subgraphs


# model
## orgin model
class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all: Tensor, device: torch.device,
                  subgraph_loader: NeighborLoader) -> Tensor:

        pbar = tqdm(total=len(subgraph_loader) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.node_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                x = x[:batch.batch_size]
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
                pbar.update(1)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all

## replica model
### EdgePooling
import logging
import random
import time
from collections import namedtuple

import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import softmax
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Set2Set
from torch_geometric.nn import global_add_pool


# import MotiFiesta.utils.learning_utils
# from ..utils.learning_utils import *

class EdgePooling(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.

    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.

    To duplicate the configuration from the "Towards Graph Pooling by Edge
    Contraction" paper, use either
    :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0`.

    To duplicate the configuration from the "Edge Contraction Pooling for
    Graph Neural Networks" paper, set :obj:`dropout` to :obj:`0.2`.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

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
        # self.conv = torch_geometric.nn.GATConv(2 * in_channels, in_channels)
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
        # self.transform_activate = torch.nn.ReLU()
        # self.transform_2 = torch.nn.Linear(out_channels, out_channels)
        # self.transform_activate = torch.nn.Sigmoid()

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
    def merge_edge_sum(x, edge_index):
        X = x[torch.flatten(edge_index.T)]
        batch = torch.arange(0, len(edge_index[0])).repeat_interleave(2)
        batch = batch.to(X.device)
        return global_add_pool(X, batch)

    def forward(self, x, edge_index, batch, hard_embed=False, dummy=False):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.

        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        # carlos: do one conv operation before scoring
        # x = x.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # computes features for each edge-connected node pair, normally just
        # sum pool
        # firt transform merged embeddings
        # logging.info(f"EdgePool: Before compute node feature {time.time()}")
        x_merged = self.edge_merge(x, edge_index)
        x_merged = self.transform(x_merged)
        # x_merged = self.transform_activate(x_merged)
        # x_merged = self.transform_2(x_merged)
        # x_merged = self.transform_activate(x_merged)

        # compute features for each node with itself in case node is not pooled
        # logging.info(f"EdgePool: Before compute self node feature {time.time()}")
        e_ind_self = torch.tensor([list(range(len(x))), list(range(len(x)))])
        x_merged_self = self.edge_merge(x, e_ind_self)
        x_merged_self = self.transform(x_merged_self)
        # x_merged_self = self.transform_activate(x_merged_self)
        # x_merged_self = self.transform_2(x_merged_self)

        # compute scores for each edge
        # logging.info(f"EdgePool: Before compute e's score {time.time()}")
        e = self.score_net(x_merged).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0), batch)

        if dummy:
            e = torch.full(e.shape, .5, dtype=torch.float32)

        x_new, edge_index, batch, unpool_info = self.__merge_edges__(
            x, edge_index, batch, e, x_merged, x_merged_self)

        # return {'new_graph': {'x_new': x_new, 'e_ind_new': edge_index, 'batch_new': batch, 'unpool': unpool_info},
        #         'internals': {'x_merged': x_merged, 'x_merged_self': x_merged_self, 'edge_scores': e}
        #         }
        return ((x_new,edge_index,batch,unpool_info),(x_merged, x_merged_self, e))
        # return x_new

    def __merge_edges__(self, x, edge_index, batch, edge_score, x_merged, x_merged_self):
        nodes_remaining = set(range(x.size(0)))
        cluster = torch.empty_like(batch, device=x.device)
        edge_argsort = torch.argsort(edge_score, descending=True)

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []

        # carlos : start building the new x
        emb_cat = []

        # edge_score_norm = torch.nn.Softmax()(edge_score.detach())
        merge_count = 0

        # edge_index_cpu = edge_index.cpu()
        edge_index_cpu = edge_index
        for edge_idx in edge_argsort.tolist():
            source = edge_index_cpu[0, edge_idx].item()
            # carlos
            r = random.random()
            # if r > edge_score[edge_idx] - .5:
            if r > edge_score[edge_idx]:
            # if r > edge_score_norm[edge_idx]:
                # print("skipped ", edge_score[edge_idx])
                continue

            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            # print(f"merged score: ", edge_score[edge_idx])
            merge_count += 1
            # emb_cat.append(torch.cat((x[source], x[target])))
            emb_cat.append(x_merged[edge_idx])
            new_edge_indices.append(edge_idx)

            cluster[source] = i
            nodes_remaining.remove(source)

            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)

            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            # emb_cat.append(torch.cat((x[node_idx], x[node_idx])))
            emb_cat.append(x_merged_self[node_idx])
            i += 1

        # carlos
        # new_x = x.new_zeros((len(emb_cat), len(emb_cat[0])), dtype=torch.float)
        new_x = torch.vstack(emb_cat)
        # for ind, emb in enumerate(emb_cat):
        #     new_x[ind] = emb
        # print(new_x[:5,:5])
        # exit()
        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0: 
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])

        # scale embedding with score
        # might want to take this out since we use the embedding in
        # reconstruction
        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)
        
        # I added this.. for some reason we were creating self loops..
        new_edge_index, _ = remove_self_loops(new_edge_index)
        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        # i added e to the output (edge scores for original graph)
        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch,
                                              new_edge_score=new_edge_score,
                                              old_edge_score=edge_score)

        return new_x, new_edge_index, new_batch, unpool_info

    def unpool(self, x, unpool_info):
        r"""Unpools a previous edge pooling step.

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (Tensor): The node features.
            unpool_info (unpool_description): Information that has
                been produced by :func:`EdgePooling.forward`.

        Return types:
            * **x** *(Tensor)* - The unpooled node features.
            * **edge_index** *(LongTensor)* - The new edge indices.
            * **batch** *(LongTensor)* - The new batch vector.
        """

        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.in_channels)


### motifiesta
import random
from collections import defaultdict
from collections import Counter
import time
import logging

from tqdm import tqdm
import torch
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from torch.nn.functional import normalize
from sklearn.neighbors import KernelDensity as KDE
from sklearn.neighbors import KDTree
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma
import networkx as nx
import matplotlib.pyplot as plt

import context
# from MotiFiesta.src.edge_pool import EdgePooling
from MotiFiesta.utils.subgraph_similarity import build_wwl_K
from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.utils.graph_utils import get_edge_subgraphs, update_merge_graph, update_spotlights


class MotiFiesta(torch.nn.Module):
    """GCN model that iteratively applies edge contraction and computes node embeddings.
    """
    def __init__(self,
                 n_features=16,
                 dim=32,
                 steps=5,
                 conv=None,
                 hard_embed=False,
                 pool_dummy=None,
                 merge_method='sum',
                 global_pool=global_add_pool,
                 edge_score_method='sigmoid'
                 ):
        """

        :param n_features: number of input features (default=16)
        :type n_features: int
        :param dim: hidden dimension size (default=32)
        :type dim: int
        :param steps: number of contration steps to run (default=5)
        :param steps: int
        :param conv: whether to apply a GCN aggregation to get node embeddings.
         should be a list conv[s] = 0 if no conv 1 if conv at each step (default=None)
        :type conv: bool
        :param ged_cache: whether to cache GED values for speedup
        (default=True)
        :type ged_cache: bool

        """
        super(MotiFiesta, self).__init__()

        self.steps = steps
        self.n_features = n_features
        self.hidden_dim = dim
        self.hard_embed = hard_embed
        self.edge_score_method = edge_score_method
        self.merge_method = merge_method
        self.hard_embed = hard_embed

        self.layers = self.build_layers()


    def build_layers(self):
        """ Construct model's layers. The model consists of stacked embedding
        and pooling layers. Embedding and pooling layers are side by side. The
        embedding layers takes the graph at time $t$ as input and outputs a
        node embedding for each node. The pooling layer takes the graph at time
        $t$ as input and returns a probability for each edge.
        """
        layers = []
        layers.append(EdgePooling(self.n_features,
                                  self.hidden_dim,
                                  edge_score_method=self.edge_score_method,
                                  merge_method=self.merge_method
                                  ))
        for s in range(self.steps):
            layers.append(EdgePooling(self.hidden_dim,
                                      self.hidden_dim,
                                      edge_score_method=self.edge_score_method,
                                      merge_method=self.merge_method
                                      )
                          )

        return torch.nn.ModuleList(layers)

    def forward(self, x, edge_index, batch, dummy=False, x_null=None, e_null=None):
        """One forward pass applies the model over all steps (all the way up).
        This function computes embeddings and probabilities and constructs two
        data structures which store the history of pooling operations.

        Returned embeddings, probabilities, and edge lists are lists of lists
        where `embeddings[t]` is a list of node embeddings at time `t`.

        The `merge_info` object is a dictionary with two keys, `tree` and
        `spotlights`. The `tree` key contains a dictionary where
        `tree[t]` contains merging info at time `t`. `tree[t][c]`
        contains the edges (pairs of nodes) assigned to `c` at time `t`. The
        `spotlights` dictionary contains the set of nodes in the base graph that each
         merged node represents.

        :param x: node features
        :param edge_index: list of edges
        :param batch: batching tensor

        :return: node embeddings, contraction probabilities, edgelists, merge info
        """
        merge_tree = {}
        spotlights = {}
        nodes = list(range(len(x)))
        merge_tree[0] = {n:set({}) for n in nodes}
        spotlights[0] = {n:{n} for n in nodes}

        edge_index_initial = edge_index
        nodes_initial = nodes
        # keep track of embeddings, edge scores, edge_index at each step
        xx, ee, pp = [], [], []
        batches = []
        internals = []

        for t, layer in enumerate(self.layers):

            if len(edge_index[0]) < 1:
                break
            
            out = layer(x, edge_index, batch, hard_embed=self.hard_embed, dummy=dummy)
            # # record graph state at time t
            
            xx.append(x)
            ee.append(edge_index)
            batches.append(batch)
            # pp.append(out['internals']['edge_scores'])
            pp.append(out[1][2])
            # print(pp)
            # internals.append(out[1])
            # record merging events which determine state at t+1
            # update_merge_graph(merge_tree, out['new_graph']['unpool'].cluster, t+1)
            # update_spotlights(spotlights, out['new_graph']['unpool'].cluster, t+1)

            # reset the graph to t+1 state
            # edge_index = out['new_graph']['e_ind_new']
            # x = out['new_graph']['x_new']
            x = out[0][0]
            xx.append(x)
            print(x)
            # x = out
            break
            batch = out['new_graph']['batch_new']
            logging.info(f"Forward {t}: End of reset {time.time()}")

        merge_info = {'tree': merge_tree, 'spotlights': spotlights}
        # return x
        print(pp[0].shape)
        pp = torch.vstack(pp)
        print(pp.shape)
        return xx #, pp, ee, batches, merge_info, internals
        # return x, pp, ee, batches, merge_info, internals


    @staticmethod
    def kde(X, X_ref, h=1):
        """ Compute gaussian KDE estimate for each entry in
        X with respect to X_ref
        """
        kde = KDE(kernel='gaussian', bandwidth=h).fit(X_ref.detach().numpy())
        f = kde.score_samples(X.detach().numpy())
        f = torch.tensor(f, dtype=torch.float32)
        return torch.exp(f)

    @staticmethod
    def knn_density(X, X_ref, volume=False, epsilon=1e-5, k=50):
        """
        \hat{f}_{X, k}(x) = \frac{k}{N} \times \frac{1}{V^{d} R_{k}(x)}$
        where $R_{d}(x)$ is the radius of a $d$-dimensional sphere
        (i.e. the distance to the $k$-th nearest neighbor) with
        volume $V^{d} = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)}$
        and $\Gamma(x)$ is the Gamma function.
        """
        d = X.shape[1]
        N = X_ref.shape[0]
        knn = KDTree(X_ref.cpu().detach().numpy())

        R,_ = knn.query(X.cpu().detach().numpy(), k=k)
        # R /= min(R) + epsilon
        R = R[:,k-1]
        if volume:
            V = ((np.pi**(d/2)) / gamma(d/2 +1)) * (R**d)
            f_hat = (k / N) * (1 / V)
        else:
            # f_hat = 1/(R + epsilon)
            f_hat = R

        f_hat = torch.tensor(f_hat, dtype=torch.float32, requires_grad=False)
        f_hat = f_hat.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        return f_hat

    @staticmethod
    def min_density(X, X_ref, epsilon=1e-5, k=10):
        """
        Estimate density as distance to nearest point.
        """
        d = torch.cdist(X, X_ref)
        return 1/d.min(dim=1)[0]


def matrix_cosine(a, b, eps=1e-8):
    """
    Pairwise cosine of embeddings
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class HardEmbedder(torch.nn.Module):
    def __init__(self, out_dim):
        super(HardEmbedder, self).__init__()
        self.out_dim = out_dim
        # self.linear_relu_stack = torch.nn.Sequential(
            # torch.nn.Linear(50, out_dim),
            # torch.nn.ReLU(),
        # )

    def forward(self, t, spotlights, edge_index_initial, nodes_initial):
        """ For now just compute a degree histogram.
        """
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes_initial)))
        G.add_edges_from(zip(*edge_index_initial.cpu().detach().numpy()))

        edge_index_initial.to(get_device())

        # nx.draw(G)
        # plt.show()

        def spotlight_graph(node):
            return G.subgraph(spotlights[t][node]).copy()

        # embeddings = torch.Tensor((len(spotlights[t]), 50))
        embeddings = []
        for pool_node in range(len(spotlights[t])):
            subg = spotlight_graph(pool_node)
            if t == 0:
                degs = Counter((G.degree(n) for n in subg.nodes()))
            else:
                degs = Counter((subg.degree(n) for n in subg.nodes()))

            deg_hist = [degs[ind] for ind in range(self.out_dim)]
            # if t > 0:
                # nx.draw(subg)
                # plt.show()
            embeddings.append(torch.Tensor(deg_hist))
            pass

        embeddings = torch.stack(embeddings)
        return embeddings

def plot_K(K_true, K_pred):
    fig, ax = plt.subplots(1, 2)
    sns.heatmap(K_true.detach().numpy(), vmin=0, vmax=1, ax=ax[0])
    sns.heatmap(K_pred.detach().numpy(), vmin=0, vmax=1, ax=ax[1])
    plt.show()
    pass


def rec_loss_fn(xx,ee,spotlights,graphs,batch,node_feats,internals,
            steps=5,
            num_nodes=20,
            draw=False):
    """Compute reconstruction loss at all coarsening
    levels.
    The loss function for a pair of embeddings z_1, z_2 and graph kernel K is:
    L = ((x_1 - x_2)^2  - K(g_1, g_2))^2
    where g_1 is the spotlight of node 1
    Here we supervise the embedding for pairs of nodes.
    """
    loss = 0
    def get_edge_subgraphs_by_level(level, ee, internals, spotlights,graphs, node_feats, batch):
        x = internals[level]['x_merged']
        extract_start = time.time()
        subgraphs, node_features = get_edge_subgraphs(ee[level],
                                                        spotlights,
                                                        level,
                                                        graphs,
                                                        node_feats,
                                                        batch,)
        # embedding are normalized so taking distance is equivalent to cosine
        K_predict = matrix_cosine(x[:num_nodes], x[:num_nodes])
        K_true = build_wwl_K(subgraphs[:num_nodes], node_features[:num_nodes])
        K_true = K_true.to(x.device)
        l = torch.nn.MSELoss()(K_predict, K_true)
        return l

    loss = sum([get_edge_subgraphs_by_level(level, ee,internals, spotlights,graphs, node_feats, batch) 
        for level in range(len(xx))])

    return loss / steps


def freq_loss_fn(internals_pos,internals_neg,pp,
                estimator='knn',
                beta=1,
                lam=1,
                steps=3,
                volume=False,
                k=30,):
    """ Penalize embeddings that are close to randos or sparse.
    """
    tot_loss = 0
    for t in range(len(pp)):
        x_pos = internals_pos[t]['x_merged']
        x_neg = internals_neg[t]['x_merged']
        s = pp[t].cpu()
        # if estimator == 'knn':
        density_pos = distance_density(x_pos, x_pos, k=k)
        density_neg = distance_density(x_pos, x_neg, k=k)

        f_pos = density_pos.view(-1, 1).squeeze().cpu()
        f_neg  = density_neg.view(-1, 1).squeeze().cpu()

        # f_pos and f_neg are [0, 1]. When density is high f -> 0, 1 else
        # f_pos - f_neg -> -1 with motifs (f_p = 0, f_n = 1)
        # f_pos - f_neg -> 1 with non-motifs (f_p = 1, f_n=0)
        l = (-1 * s * torch.exp(-1 * beta * (f_pos - f_neg))).mean()

        reg_term = lam * s.pow(2.0).mean()
        l += reg_term

        tot_loss += l

    tot_loss /= steps
    return tot_loss


def distance_density(X, X_ref, k=20):
    """ Returns distance to kth nearest neighbor in batch
    """
    N = X_ref.shape[0]
    # edited by roccc: add modification on k value
    k = min(k, N)
    knn = KDTree(X_ref.cpu().detach().numpy())
    R,_ = knn.query(X.cpu().detach().numpy(), k=k)
    R = R[:,k-1]
    return torch.tensor(R, dtype=torch.float32, requires_grad=False)



# run
def run(rank, world_size, loader_train):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # data = dataset[0]
    # data = data.to(rank, 'x', 'y')  # Move to device for faster feature fetch.

    # Split training indices into `world_size` many chunks:
    # train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    # train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    # kwargs = dict(batch_size=1024, num_workers=4, persistent_workers=True)
    # train_loader = NeighborLoader(data, input_nodes=train_idx,
    #                               num_neighbors=[25, 10], shuffle=True,
    #                               drop_last=True, **kwargs)
    train_loader = loader_train

    # if rank == 0:  # Create single-hop evaluation neighbor loader:
    #     subgraph_loader = NeighborLoader(copy.copy(data), num_neighbors=[-1],
    #                                      shuffle=False, **kwargs)
    #     # No need to maintain these features during evaluation:
    #     del subgraph_loader.data.x, subgraph_loader.data.y
    #     # Add global node index information:
    #     subgraph_loader.data.node_id = torch.arange(data.num_nodes)

    torch.manual_seed(12345)
    # model = SAGE(2, 256, 8).to(rank)
                    # 'dim': args.dim,
                    # 'steps': args.steps,
                    # 'pool_dummy': args.pool_dummy,
                    # 'edge_score_method': args.score_method,
                    # 'merge_method': args.merge_method,
                    # 'hard_embed': args.hard_embed,
    model = MotiFiesta(n_features=2, dim=8,steps=5,pool_dummy=False, edge_score_method='sigmoid',merge_method='sum',hard_embed=None).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 21):
        model.train()
        for (batch_pos, batch_neg) in train_loader:
            optimizer.zero_grad()
            graphs_pos = to_graphs(batch_pos)
            batch_pos = batch_pos.to(rank)

            x_pos, edge_index_pos = batch_pos.x.float(), batch_pos.edge_index
            # xx_pos, pp_pos, ee_pos,_, merge_info_pos, internals_pos = model(x_pos,
            #                                                                 edge_index_pos,
            #                                                                 batch_pos.batch)
            # xx_pos = xx_pos[-1]
            xx_pos = model(x_pos, edge_index_pos,batch_pos.batch)
            xx_pos = xx_pos[-1]
            # loss

            y = torch.rand(size=xx_pos.shape, device=xx_pos.device)
            loss = F.mse_loss(xx_pos, y)
            
            # loss = rec_loss_fn(
            #     xx_pos,
            #     ee_pos,
            #     merge_info_pos['spotlights'],
            #     graphs_pos,
            #     batch_pos.batch,
            #     x_pos,
            #     internals_pos,
            #     draw=False)
            loss.backward()
            optimizer.step()

        dist.barrier()

        if rank == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        # if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
        #     model.eval()
        #     with torch.no_grad():
        #         out = model.module.inference(data.x, rank, subgraph_loader)
        #     res = out.argmax(dim=-1) == data.y.to(out.device)
        #     acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
        #     acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
        #     acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
        #     print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        # dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    # dataset
    ## origin data
    # dataset = Reddit('Reddit')

    ## new data
    from MotiFiesta.src.loading import get_loader
    loader_train, loader_test, num_features = get_loader(
                        name='ZINC_1k',
                        batch_size=128,
                        num_workers=1,
                        local_rank=-1,
                        )

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size, loader_train), nprocs=world_size, join=True)

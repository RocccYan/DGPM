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

import sys
sys.path.append("/mnt/workspace/graph_pretrain")
from MotiFiesta.src.edge_pool import EdgePooling
from MotiFiesta.utils.subgraph_similarity import build_wwl_K
from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.utils.graph_utils import get_edge_subgraphs, update_merge_graph, update_spotlights


# roc, refer from [2020Strategy], for chem datasets
num_atom_type = 120 
num_chirality_tag = 3

num_bond_type = 6 
num_bond_direction = 3 


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
                 edge_score_method='sigmoid',
                 chem=False,
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
        # roc, edge attr -> edge embeddings
        self.chem = chem
        
        self.layers = self.build_layers()

    def build_layers(self):
        """ Construct model's layers. The model consists of stacked embedding
        and pooling layers. Embedding and pooling layers are side by side. The
        embedding layers takes the graph at time $t$ as input and outputs a
        node embedding for each node. The pooling layer takes the graph at time
        $t$ as input and returns a probability for each edge.
        """
        # roc, add edge_attr into aggregation
        if self.chem:
            # encode x

            self.x_embedding1 = torch.nn.Embedding(num_atom_type, self.hidden_dim)
            self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, self.hidden_dim)
            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

            # encode edge
            self.edge_embedding1 = torch.nn.Embedding(num_bond_type, self.hidden_dim)
            self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, self.hidden_dim)
            torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
            
            self.n_features = self.hidden_dim
        
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
    
    def forward(self, x, edge_index, batch, dummy=False, x_null=None, e_null=None, edge_attr=None):
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
        # roc, encode chem if necessary
        edge_emb = None
        # print(x)
        # print(edge_attr)
        if self.chem:
            x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
            edge_emb = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
            print("chem ENCODED!")
            # print(x)
            # print(edge_emb)
            # # exit()

        # keep track of embeddings, edge scores, edge_index at each step
        xx, ee, pp = [], [], []
        batches = []
        internals = []

        for t, layer in enumerate(self.layers):

            if len(edge_index[0]) < 1:
                break
            if x.dtype != torch.float:
                x = x.float()
            out = layer(x, edge_index, batch, edge_emb=edge_emb, hard_embed=self.hard_embed, dummy=dummy)
            # record graph state at time t
            ee.append(edge_index)
            xx.append(x)
            pp.append(out['internals']['edge_scores'])
            batches.append(batch)
            internals.append(out['internals'])
            # logging.info(f"Forward {t}: End of append {time.time()}")

            # record merging events which determine state at t+1
            update_merge_graph(merge_tree, out['new_graph']['unpool'].cluster, t+1)
            update_spotlights(spotlights, out['new_graph']['unpool'].cluster, t+1)
            # logging.info(f"Forward {t}: End of update {time.time()}")

            # reset the graph to t+1 state
            edge_index = out['new_graph']['e_ind_new']
            x = out['new_graph']['x_new']
            batch = out['new_graph']['batch_new']
            edge_emb = out['new_graph']['edge_emb_new']
            # logging.info(f"Forward {t}: End of reset {time.time()}")

        merge_info = {'tree': merge_tree, 'spotlights': spotlights}

        return xx, pp, ee, batches, merge_info, internals
    
    def encode_graph_level(self, data, encode_mode="cat", edge_attr=None):

        embs,_,_,batches,merge_info,_  = self.forward(data.x,
                                                      data.edge_index,
                                                      data.batch,
                                                      edge_attr=edge_attr)
        # pooling by layers, with new batch by layers
        if encode_mode == 'single_layer':
            # roc, take the last layer
            emb = global_add_pool(embs[-1], batches[-1])
            return emb
        else:
            x_layers = []
            for l in range(1, len(embs)):
                X_l = embs[l]
                batch = batches[l]
                # when pool graph for each level, add means summarization of the whole graph
                h = global_add_pool(X_l, batch)
                # print(h.shape)
                x_layers.append(h)
            # print(x_layers)
            # sum, increased as layers, try mean
            if encode_mode == 'sum':
                return torch.sum(torch.stack(x_layers), dim=0)
            else:
                # cat
                return torch.cat(x_layers, dim=1)

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
        # TODO: features for wwl should be continous.
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

        density_pos = distance_density(x_pos, x_pos, k=k)
        density_neg = distance_density(x_pos, x_neg, k=k)

        f_pos = density_pos.view(-1, 1).squeeze()
        f_neg  = density_neg.view(-1, 1).squeeze()

        # f_pos and f_neg are [0, 1]. When density is high f -> 0, 1 else
        # f_pos - f_neg -> -1 with motifs (f_p = 0, f_n = 1)
        # f_pos - f_neg -> 1 with non-motifs (f_p = 1, f_n=0)
        l = (-1 * s * torch.exp(-1 * beta * (f_pos - f_neg))).mean()
        # roc, sigmoid + sum
        # logging.info(f"Loss Sigmoid + sum.")
        # l = (s * torch.exp(beta * (torch.sigmoid(f_pos - f_neg)))).sum()
        logging.info(f"Loss before reg_term: {l}")
        reg_term = lam * s.pow(2.0).mean()
        # reg_term = lam * s.pow(2.0).sum()
        l += reg_term
        logging.info(f"Loss After reg_term: {l}")
        tot_loss += l
    # exit()
    tot_loss /= steps
    logging.info(f"Batch Mot Loss: {l}")
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
    # return torch.tensor(R, dtype=torch.float32, requires_grad=False)
    return torch.tensor(R, dtype=torch.float32, requires_grad=True)


class GraphClassifier(torch.nn.Module):
    def __init__(self,encoder_motif=None, encoder_node=None, emb_dim=0,num_tasks=0, ):
        super(GraphClassifier, self).__init__()
        self.encoder_motif = encoder_motif
        self.encoder_node = encoder_node
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        # define classification layer
        self.classification = torch.nn.Linear(self.emb_dim, self.num_tasks)
    
    def forward(self, data, mode='motif'):
        if mode == 'motif':
            emb = self.encoder_motif.encode_graph_level(data)
        else:
            # TODO
            pass
        return self.classification(emb)
        

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    pass

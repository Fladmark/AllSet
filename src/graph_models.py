""""
Code from: https://github.com/ycq091044/LEGCN/blob/master/src/models.py.
Some additions made.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layers import GraphConvolution, GraphAttentionLayer, SpGraphAttentionLayer
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout, Embedding
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter_sum

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.layers = [self.gc1, self.gc2]
        self.dropout = dropout

    def forward(self, x, adj, PvT):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        # x = F.relu(self.gc3(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)
        x = torch.spmm(PvT, x)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.alpha = alpha
        self.nheads = nheads
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, PvT):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = torch.spmm(PvT, x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.attentions = [GraphAttentionLayer(self.nfeat, self.nhid, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(self.nhid * self.nheads, self.nclass, dropout=self.dropout, alpha=self.alpha, concat=False)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha=0.2, nheads=8):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.alpha = alpha
        self.nheads = nheads
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj, PvT):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = torch.spmm(PvT, x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.attentions = [SpGraphAttentionLayer(self.nfeat,
                                                 self.nhid,
                                                 dropout=self.dropout,
                                                 alpha=self.alpha,
                                                 concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(self.nhid * self.nheads,
                                             self.nclass,
                                             dropout=self.dropout,
                                             alpha=self.alpha,
                                             concat=False)


class GINq(torch.nn.Module):
    """GIN"""

    def __init__(self, nfeat, dim_h, nclass):
        super(GIN, self).__init__()

        self.nfeat = nfeat
        self.dim_h = dim_h
        self.nclass = nclass

        self.conv1 = GINConv(
            Sequential(Linear(nfeat, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, nclass)

    def forward(self, x, edge_index, PvT, batch=1):
        # Node embeddings
        print(edge_index.shape)
        edge_index = edge_index.type(torch.LongTensor)

        #print((edge_index @ PvT).shape)

        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        # h1 = global_add_pool(h1, batch)
        # h2 = global_add_pool(h2, batch)
        # h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)
        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return F.log_softmax(h, dim=1)

    def reset_parameters(self):
        self.conv1 = GINConv(
            Sequential(Linear(self.nfeat, self.dim_h),
                       BatchNorm1d(self.dim_h), ReLU(),
                       Linear(self.dim_h, self.dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(self.dim_h, self.dim_h), BatchNorm1d(self.dim_h), ReLU(),
                       Linear(self.dim_h, self.dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(self.dim_h, self.dim_h), BatchNorm1d(self.dim_h), ReLU(),
                       Linear(self.dim_h, self.dim_h), ReLU()))
        self.lin1 = Linear(self.dim_h * 3, self.dim_h * 3)
        self.lin2 = Linear(self.dim_h * 3, self.nclass)


class GINLayer(nn.Module):
    """A single GIN layer, implementing MLP(AX + (1+eps)X)"""

    def __init__(self, in_feats: int, out_feats: int, hidden_dim: int, eps: float = 0.0):
        super(GINLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

        # ============ YOUR CODE HERE =============
        # epsilon should be a learnable parameter
        self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        # =========================================
        self.linear1 = nn.Linear(self.in_feats, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.out_feats)

    def forward(self, x, adj_sparse):
        # ============ YOUR CODE HERE =============
        # aggregate the neighbours as in GIN: (AX + (1+eps)X)
        print(adj_sparse.shape, x.shape)
        x = torch.mm(adj_sparse, x) + (1 + self.eps) * x
        # project the features (MLP_k)
        out = self.linear2(F.relu(self.linear1(x)))
        # ========================================
        return out

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()


class GIN(nn.Module):
    """
    A Graph Neural Network containing GIN layers
    as in https://arxiv.org/abs/1810.00826
    The readout function used to obtain graph-lvl representations
    aggregate pred from multiple layers (as in JK-Net)

    Args:
    input_dim (int): Dimensionality of the input feature vectors
    output_dim (int): Dimensionality of the output softmax distribution
    num_layers (int): Number of layers
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, eps=0.0):
        super(GIN, self).__init__()

        self.num_layers = num_layers  # please select num_layers >=1
        # nodes in ZINC dataset are characterised by one integer (atom category)
        # we will create embeddings from the categorical features using nn.Embedding

        self.embed_x = Linear(input_dim, hidden_dim)

        # ============ YOUR CODE HERE =============
        # should be the same as before (an nn.ModuleList of GINLayers)
        self.layers = [GINLayer(hidden_dim, hidden_dim, hidden_dim, eps) for _ in range(num_layers - 1)]
        self.layers += [GINLayer(hidden_dim, output_dim, hidden_dim, eps)]
        self.layers = nn.ModuleList(self.layers)
        # layer to compute prediction from the concatenated intermediate representations
        # self.pred_layers = torch.nn.ModuleList()

        self.pred_layers = Linear(num_layers * hidden_dim, output_dim)
        # =========================================

    def forward(self, x, adj, PvT):
        adj_sparse = adj

        x = self.embed_x(x)

        batch = torch.zeros(adj.shape[1]).type(torch.LongTensor)

        catter = scatter_sum(x, batch, dim=0)

        # ============ YOUR CODE HERE =============
        # perform the forward pass with the new readout function

        for i in range(self.num_layers - 1):
            x = self.layers[i](x.type(torch.LongTensor), adj_sparse)
            x = F.relu(x)

            ssum = scatter_sum(x, batch, dim=0)
            catter = torch.cat((catter, ssum), dim=1)

        y_hat = self.pred_layers(catter).flatten()

        # =========================================
        # return also the final node embeddings (for visualisations)
        return y_hat, x

    def reset_parameters(self):
        self.embed_x.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.pred_layers.reset_parameters()
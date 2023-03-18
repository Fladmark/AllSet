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
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter_sum
from torch_geometric.utils import add_self_loops, degree
import scipy.sparse as sp
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.layers = [self.gc1, self.gc2]
        self.dropout = dropout
        print("nfeat: " + str(nfeat))
        print("nclass: " + str(nclass))
        print("nhid: " + str(nhid))

    def forward(self, x, adj, PvT):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        print("start")
        print(x.shape)

        # x = F.relu(self.gc3(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)
        print(x.shape)
        x = torch.spmm(PvT, x)
        print(x.shape)

        print(F.log_softmax(x, dim=1).shape)

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
        x = x.float() # fix
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
        #print(x.shape)

        batch = torch.zeros(adj.shape[1]).type(torch.LongTensor)

        catter = scatter_sum(x, batch, dim=0)
        #print(catter.shape)

        # ============ YOUR CODE HERE =============
        # perform the forward pass with the new readout function

        for i in range(self.num_layers - 1):
            x = self.layers[i](x.type(torch.LongTensor), adj_sparse)
            x = F.relu(x)

            ssum = scatter_sum(x, batch, dim=0)
            catter = torch.cat((catter, ssum), dim=1)
        #print(x.shape)

        y_hat = self.pred_layers(catter).flatten()
        #print(y_hat.shape)

        # =========================================
        # return also the final node embeddings (for visualisations)
        #return y_hat#, x
        x = torch.spmm(PvT, x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.embed_x.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.pred_layers.reset_parameters()

# Not used
def sparse_adjacency_matrix_to_edge_index(sparse_adj_matrix):
    sparse_adj_matrix = sparse_adj_matrix.tocoo()
    row, col = sparse_adj_matrix.row, sparse_adj_matrix.col
    edge_index = torch.stack((torch.tensor(row), torch.tensor(col)), dim=0)
    return edge_index

def sparse_tensor_to_edge_index(sparse_adj_tensor):
    row, col = sparse_adj_tensor.coalesce().indices()
    edge_index = torch.stack((row, col), dim=0)
    return edge_index

class MPNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNConv, self).__init__(aggr='add')  # Aggregates messages by addition
        self.lin = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        return x_j

    def update(self, aggr_out):
        return aggr_out

class MPNNNodeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MPNNNodeClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.conv1 = MPNNConv(input_dim, hidden_dim)
        self.conv2 = MPNNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.layers = [self.conv1, self.conv2, self.fc]

    def forward(self, x, adj, PvT):
        #x, edge_index = data.x, data.edge_index
        edge_index = sparse_tensor_to_edge_index(adj)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        x = torch.spmm(PvT, x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1 = MPNNConv(self.input_dim, self.hidden_dim)
        self.conv2 = MPNNConv(self.hidden_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        #for layer in self.layers:
        #    layer.reset_parameters()



####


class ChebGraphConv(nn.Module):
    def __init__(self, K, in_features, out_features, bias):
        super(ChebGraphConv, self).__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(K, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gso):
        # Chebyshev polynomials:
        # x_0 = x,
        # x_1 = gso * x,
        # x_k = 2 * gso * x_{k-1} - x_{k-2},
        # where gso = 2 * gso / eigv_max - id.

        cheb_poly_feat = []
        if self.K < 0:
            raise ValueError('ERROR: The order of Chebyshev polynomials shoule be non-negative!')
        elif self.K == 0:
            # x_0 = x
            cheb_poly_feat.append(x)
        elif self.K == 1:
            # x_0 = x
            cheb_poly_feat.append(x)
            if gso.is_sparse:
                # x_1 = gso * x
                cheb_poly_feat.append(torch.sparse.mm(gso, x))
            else:
                if x.is_sparse:
                    x = x.to_dense
                # x_1 = gso * x
                cheb_poly_feat.append(torch.mm(gso, x))
        else:
            # x_0 = x
            cheb_poly_feat.append(x)
            if gso.is_sparse:
                # x_1 = gso * x
                cheb_poly_feat.append(torch.sparse.mm(gso, x))
                # x_k = 2 * gso * x_{k-1} - x_{k-2}
                for k in range(2, self.K):
                    cheb_poly_feat.append(torch.sparse.mm(2 * gso, cheb_poly_feat[k - 1]) - cheb_poly_feat[k - 2])
            else:
                if x.is_sparse:
                    x = x.to_dense
                # x_1 = gso * x
                cheb_poly_feat.append(torch.mm(gso, x))
                # x_k = 2 * gso * x_{k-1} - x_{k-2}
                for k in range(2, self.K):
                    cheb_poly_feat.append(torch.mm(2 * gso, cheb_poly_feat[k - 1]) - cheb_poly_feat[k - 2])

        feature = torch.stack(cheb_poly_feat, dim=0)
        if feature.is_sparse:
            feature = feature.to_dense()
        cheb_graph_conv = torch.einsum('bij,bjk->ik', feature, self.weight)

        if self.bias is not None:
            cheb_graph_conv = torch.add(input=cheb_graph_conv, other=self.bias, alpha=1)
        else:
            cheb_graph_conv = cheb_graph_conv

        return cheb_graph_conv

    def extra_repr(self) -> str:
        return 'K={}, in_features={}, out_features={}, bias={}'.format(
            self.K, self.in_features, self.out_features, self.bias is not None
        )


class ChebyNet(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, K_order, K_layer, droprate, gso):
        super(ChebyNet, self).__init__()
        self.cheb_graph_convs = nn.ModuleList()
        self.K_order = K_order
        self.K_layer = K_layer
        self.cheb_graph_convs.append(ChebGraphConv(K_order, n_feat, n_hid, enable_bias))
        for k in range(1, K_layer-1):
            self.cheb_graph_convs.append(ChebGraphConv(K_order, n_hid, n_hid, enable_bias))
        self.cheb_graph_convs.append(ChebGraphConv(K_order, n_hid, n_class, enable_bias))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.gso = gso

    def forward(self, x, adj, PvT):
        for k in range(self.K_layer-1):
            x = self.cheb_graph_convs[k](x, self.gso)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.cheb_graph_convs[-1](x, self.gso)
        x = torch.spmm(PvT, x)
        x = self.log_softmax(x)

        return x

    def reset_parameters(self):
        for layer in self.cheb_graph_convs:
            layer.reset_parameters()
""""
Code from: https://github.com/ycq091044/LEGCN/blob/master/src/models.py.
Some additions made.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layers import GraphConvolution, GraphAttentionLayer, SpGraphAttentionLayer

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
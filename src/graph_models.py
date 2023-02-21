""""
Code from: https://github.com/ycq091044/LEGCN/blob/master/src/models.py
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
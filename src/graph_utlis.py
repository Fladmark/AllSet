import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

# from LEGCN
from src.convert_datasets_to_pygDataset import dataset_Hypergraph


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = (r_mat_inv).dot(mx)
    return mx

# from LEGCN
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# from train.py - Tweaks: added adj and PvT for GCN usability
@torch.no_grad()
def evaluate_GCN(model, data, split_idx, eval_func, adj, PvT, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data.x, adj, PvT)
        out = F.log_softmax(out, dim=1)

    print(out)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out

# from train.py
def get_data(dname, feature_noise=0):
    ### Load and preprocess data ###
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom',
                        'coauthor_cora', 'coauthor_dblp',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed']

    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100',
                      'house-committees-100']

    if dname in existing_dataset:
        dname = dname
        f_noise = feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,
                                         feature_noise=f_noise,
                                         p2raw=p2raw)
        else:
            if dname in ['cora', 'citeseer', 'pubmed']:
                p2raw = '../data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = '../data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = '../data/AllSet_all_raw_data/yelp/'
            else:
                p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, root='../data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw=p2raw)

    return dataset
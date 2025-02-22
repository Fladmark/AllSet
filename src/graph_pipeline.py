from convert_datasets_to_pygDataset import dataset_Hypergraph
from expansions import line_expansion, clique_expansion, line_graph, star_expansion, lawler_expansion, line_expansion_2
from graph_utlis import normalize, sparse_mx_to_torch_sparse_tensor, evaluate_GCN, get_data
from preprocessing import rand_train_test_idx, ExtractV2E
from train import Logger, count_parameters, eval_acc

from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import graph_models
import numpy as np
import os.path as osp
import os
import scipy.sparse as sp
import graph_utlis
import sys

dname, model_name, expansion_name = sys.argv[1:]

# if os.path.exists(f'{"hyperparameter_tunning"}/{"expansion_experiments"}/{dname}_{model_name}_{expansion_name}.csv'):
#     print(dname, model_name, expansion_name)
#     print("Already exists, going next")
#     quit()


print(dname, model_name, expansion_name)

#dname = "Mushroom"
#dname = "house-committees-100"
#dname = "cora"
#dname = "zoo"
#dname = "citeseer"

dataset = get_data(dname)

print(sum(dataset.data.x))

# Hacky way of halfing the size of edge index (because dataset_Hypergraph for some reason duplicated this)
single_edge_index = [[],[]]
edge_index_overview = set()

for i in range(dataset.data.edge_index.shape[1]):
    t1 = (dataset.data.edge_index[0][i].item(), dataset.data.edge_index[1][i].item())
    t2 = (dataset.data.edge_index[1][i].item(), dataset.data.edge_index[0][i].item())
    if t1 in edge_index_overview or t2 in edge_index_overview:
        continue
    else:
        single_edge_index[0].append(dataset.data.edge_index[0][i])
        single_edge_index[1].append(dataset.data.edge_index[1][i])
        edge_index_overview.add(t1)

dataset.data.edge_index = torch.tensor(single_edge_index)

pairs = (dataset.data.edge_index.numpy().T)

# Choose expansion
#expansion_name = "line_expansion"

if expansion_name == "line_expansion":
    adj, Pv, PvT = line_expansion_2(pairs, dataset.data.y, 30, 30)
elif expansion_name == "clique_expansion":
    adj, Pv, PvT = clique_expansion(pairs, dataset.data.y)
elif expansion_name == "line_graph":
    adj, Pv, PvT = line_graph(pairs, dataset.data.y)
elif expansion_name == "star_expansion_1":
    adj, Pv, PvT = star_expansion(pairs, dataset.data.y, method=1)
elif expansion_name == "star_expansion_2":
    adj, Pv, PvT = star_expansion(pairs, dataset.data.y, method=2)
elif expansion_name == "lawler_expansion_1":
    adj, Pv, PvT = lawler_expansion(pairs, dataset.data.y, method=1)
elif expansion_name == "lawler_expansion_2":
    adj, Pv, PvT = lawler_expansion(pairs, dataset.data.y, method=2)

#adj, Pv, PvT, Pe, PeT = line_expansion(pairs, dataset.data.y, 30, 30)
#adj, Pv, PvT = line_expansion_2(pairs, dataset.data.y, 30, 30)
#adj, Pv, PvT = clique_expansion(pairs, dataset.data.y)
#adj, Pv, PvT = line_graph(pairs, dataset.data.y)
#adj, Pv, PvT = star_expansion(pairs, dataset.data.y, method=1)
#adj, Pv, PvT = star_expansion(pairs, dataset.data.y, method=2)
#adj, Pv, PvT = lawler_expansion(pairs, dataset.data.y, method=1)
#adj, Pv, PvT = lawler_expansion(pairs, dataset.data.y, method=2)

# project features to LE
dataset.data.x = torch.FloatTensor(np.array(Pv @ dataset.data.x))

# sparse back projection matrix
PvT = sparse_mx_to_torch_sparse_tensor(PvT)


runs = 5
train_prop = 0.50
valid_prop = 0.25
lr = 0.001
wd = 0
epochs = 50
hidden = 64

if dname == "cora":
    classes = 7
elif dname == "Mushroom":
    classes = 2
elif dname == "zoo":
    classes = 7
    dataset.data.y = dataset.data.y - 1
elif dname == "citeseer":
    classes = 7
elif dname == "house-committees-100":
    classes = 2
    dataset.data.y = dataset.data.y - 1

display_step = -1
feature_noise = 0
heads = 0
method = "STANDARD_GCN"


logger = Logger(runs)
criterion = torch.nn.NLLLoss()
eval_func = eval_acc

#data = ExtractV2E(dataset.data)
data = dataset.data

split_idx_lst = []
for run in range(runs):
    split_idx = rand_train_test_idx(
        data.y, train_prop=train_prop, valid_prop=valid_prop)
    split_idx_lst.append(split_idx)

#model_name = "GCN"

if model_name == "GCN":
    model = graph_models.GCN(data.x.shape[1], hidden, classes, 0)
elif model_name == "SpGAT":
    model = graph_models.SpGAT(data.x.shape[1], hidden, classes, 0)
elif model_name == "MPNN":
    model = graph_models.MPNNNodeClassifier(data.x.shape[1], hidden, classes)
elif model_name == "ChebyNet":
    gso = graph_utlis.calc_gso(adj, 'sym_norm_lap')
    gso = graph_utlis.calc_chebynet_gso(gso)
    gso = graph_utlis.cnv_sparse_mat_to_coo_tensor(gso, 'cpu')
    model = graph_models.ChebyNet(data.x.shape[1], 64, classes, enable_bias=True, K_order=2, K_layer=2, droprate=0.5, gso=gso)

#model = graph_models.GCN(data.x.shape[1], hidden, classes, 0)
#model = graph_models.GIN(data.x.shape[1], hidden, classes)
#model = graph_models.SpGAT(data.x.shape[1], hidden, classes, 0)
#model = graph_models.MPNNNodeClassifier(data.x.shape[1], hidden, classes)
#model = graph_models.ChebyNet(data.x.shape[1], 64, classes, True, 2, 2, 0.5, gso)

num_params = count_parameters(model)
model.train()
### Training loop ###
runtime_list = []
for run in tqdm(range(runs)):
    start_time = time.time()
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train']#.to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val = float('-inf')
    for epoch in range(epochs):
        # Training part
        #print(epoch)
        model.train()
        optimizer.zero_grad()

        out = model(data.x, adj, PvT)
        #out = model(data.x, data.edge_index, PvT)
        loss = criterion(out[train_idx], data.y[train_idx])

        loss.backward()
        optimizer.step()
        #         if args.method == 'HNHN':
        #             scheduler.step()
        #         Evaluation part
        result = evaluate_GCN(model, data, split_idx, eval_func, adj, PvT)
        logger.add_result(run, result[:3])
        if epoch % display_step == 0 and display_step > 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Train Loss: {loss:.4f}, '
                  f'Valid Loss: {result[4]:.4f}, '
                  f'Test  Loss: {result[5]:.4f}, '
                  f'Train Acc: {100 * result[0]:.2f}%, '
                  f'Valid Acc: {100 * result[1]:.2f}%, '
                  f'Test  Acc: {100 * result[2]:.2f}%')

    end_time = time.time()
    runtime_list.append(end_time - start_time)

    # logger.print_statistics(run)

### Save results ###
avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)
best_val, best_test = logger.print_statistics()

best_val, best_test = logger.print_statistics()
res_root = 'hyperparameter_tunning'
if not osp.isdir(res_root):
    os.makedirs(res_root)

filename = f'{"hyperparameter_tunning"}/{"expansion_experiments"}/{dname}_{model_name}_{expansion_name}.csv'
print(f"Saving results to {filename}")
with open(filename, 'a+') as write_obj:
    cur_line = f'{method}_{lr}_{wd}_{heads}'
    cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}'
    cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}'
    cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s'
    cur_line += f',{avg_time // 60}min{(avg_time % 60):.2f}s'
    cur_line += f'\n'
    write_obj.write(cur_line)

# all_args_file = f'{res_root}/all_args_{dname}_noise_{feature_noise}.csv'
# with open(all_args_file, 'a+') as f:
#     f.write(str(args))
#     f.write('\n')

print('All done! Exit python code')
quit()

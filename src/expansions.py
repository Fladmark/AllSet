"""
Code from: https://github.com/ycq091044/LEGCN/blob/master/src/LE.py
"""

import numpy as np
import torch
import scipy.sparse as sp
from itertools import combinations

from src.graph_utlis import normalize, sparse_mx_to_torch_sparse_tensor


def line_expansion(pairs, y,v_threshold=30, e_threshold=30):
    """construct line expansion from original hypergraph
    INPUT:
        - pairs <matrix>
            - size: N x 2. N means the total vertex-hyperedge pair of the hypergraph
            - each row contains the idx_of_vertex, idx_of_hyperedge
        - v_threshold: vertex-similar neighbor sample threshold
        - e_threshold: hyperedge-similar neighbor sample threshold
    Concept:
        - vertex, hyperedge: for the hypergraph
        - node, edge: for the induced simple graph
    OUTPUT:
        - adj <sparse coo_matrix>: N_node x N_node
        - Pv <sparse coo_matrix>: N_node x N_vertex
        - PvT <sparse coo_matrix>: N_vertex x N_node
        - Pe <sparse coo_matrix>: N_node x N_hyperedge
        - PeT <sparse coo_matrix>: N_hyperedge x N_node
    """
    # get # of vertices and encode them starting from 0
    uniq_vertex = np.unique(pairs[:, 0])
    #N_vertex = len(uniq_vertex)
    N_vertex = len(y)
    pairs[:, 0] = list(map({vertex: i for i, vertex in enumerate(uniq_vertex)}.get, pairs[:, 0]))

    # get # of hyperedges and encode them starting from 0
    uniq_hyperedge = np.unique(pairs[:, 1])
    N_hyperedge = len(uniq_hyperedge)
    pairs[:, 1] = list(map({hyperedge: i for i, hyperedge in enumerate(uniq_hyperedge)}.get, pairs[:, 1]))

    N_node = pairs.shape[0]
    #print(N_vertex)
    print(pairs)

    # vertex projection: from vertex to node
    Pv = sp.coo_matrix((np.ones(N_node), (np.arange(N_node), pairs[:, 0])),
                       shape=(N_node, N_vertex), dtype=np.float32)  # (N_node, N_vertex)
    # vertex back projection (Pv Transpose): from node to vertex
    #print(pairs[:, 0])
    #for val in pairs[:, 0]:
    #    print(val)
    temp = Pv.toarray()
    #print(Pv)

    weight = np.ones(N_node)
    for vertex in range(N_vertex):
        tmp = np.where(pairs[:, 0] == vertex)[0]
        # if not in an hyperedge, weight = 1. Necessary since we no longer count unique vertices
        if len(tmp) == 0:
            weight[tmp] = 1
        else:
            weight[tmp] = 1. / len(tmp)
    PvT = sp.coo_matrix((weight, (pairs[:, 0], np.arange(N_node))),
                        shape=(N_vertex, N_node), dtype=np.float32)  # (N_vertex, N_node)

    # hyperedge projection: from hyperedge to node
    Pe = sp.coo_matrix((np.ones(N_node), (np.arange(N_node), pairs[:, 1])),
                       shape=(N_node, N_hyperedge), dtype=np.float32)  # (N_node, N_hyperedge)
    # hyperedge back projection (Pe Transpose): from node to hyperedge
    weight = np.ones(N_node)
    for hyperedge in range(N_hyperedge):
        tmp = np.where(pairs[:, 1] == hyperedge)[0]
        weight[tmp] = 1. / len(tmp)
    PeT = sp.coo_matrix((weight, (pairs[:, 1], np.arange(N_node))),
                        shape=(N_hyperedge, N_node), dtype=np.float32)  # (N_node, N_hyperedge)

    # construct adj
    edges = []
    # get vertex-similar edges
    for vertex in range(N_vertex):
        position = np.where(pairs[:, 0] == vertex)[0]
        if len(position) > v_threshold:
            position = np.random.choice(position, v_threshold, replace=False)
            tmp_edge = np.array(list(combinations(position, r=2)))
            edges += list(tmp_edge)
        else:
            edges += list(combinations(position, r=2))

    # get hyperedge-similar edges
    for hyperedge in range(N_hyperedge):
        position = np.where(pairs[:, 1] == hyperedge)[0]
        if len(position) > e_threshold:
            position = np.random.choice(position, e_threshold, replace=False)
            tmp_edge = np.array(list(combinations(position, r=2)))
            edges += list(list(tmp_edge))
        else:
            edges += list(combinations(position, r=2))

    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N_node, N_node), dtype=np.float32)

    # Makes adj symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + 2.0 * sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, Pv, PvT, Pe, PeT

def clique_expansion(pairs, y):
    N_node = len(y)
    N_vertex = len(y)

    hyper_edge_dict = dict()
    # Loop over vertex-hyperedge pairs
    for pair in pairs:
        edge_id = pair[1]
        vertex_id = pair[0]
        if hyper_edge_dict.get(edge_id) == None:
            hyper_edge_dict[edge_id] = [vertex_id]
        else:
            hyper_edge_dict[edge_id] = hyper_edge_dict[edge_id] + [vertex_id]

    # Make edge list
    edges = []

    for vertices in hyper_edge_dict.values():
        for idx_1 in range(len(vertices)):
            for idx_2 in range(idx_1 + 1, len(vertices)):
                edges.append((vertices[idx_1], vertices[idx_2]))


    # Make adjacency matrix
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N_node, N_node), dtype=np.float32)

    # Makes adj symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + 2.0 * sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # vertex projection: from vertex to node
    Pv = sp.coo_matrix(torch.eye(N_node),
                       shape=(N_node, N_vertex), dtype=np.float32)  # (N_node, N_vertex)

    PvT = sp.coo_matrix(torch.eye(N_node),
                       shape=(N_node, N_vertex), dtype=np.float32)  # (N_node, N_vertex)

    return adj, Pv, PvT

def star_expansion(pairs, y, method=1):

    N_vertex = len(y)

    # Get number of hyperedges - number of new nodes to add
    uniq_hyperedge = np.unique(pairs[:, 1])
    N_hyper_edges = len(uniq_hyperedge)

    # Re-encode hyperedges
    pairs[:, 1] = list(map({hyperedge: i+N_vertex for i, hyperedge in enumerate(uniq_hyperedge)}.get, pairs[:, 1]))

    # Number of nodes in new graph
    N_node = N_vertex + N_hyper_edges

    # Get hyperedge membership information
    hyper_edge_dict = dict()
    for pair in pairs:
        edge_id = pair[1]
        vertex_id = pair[0]
        if hyper_edge_dict.get(edge_id) == None:
            hyper_edge_dict[edge_id] = {vertex_id}
        else:
            hyper_edge_dict[edge_id].add(vertex_id)

    # Add edges based on hyperedge membership
    edges = []

    # Loop over vertices
    for vertex in range(N_vertex):
        # Loop over hyperedges
        for hyperedge in hyper_edge_dict.keys():
            # If vertex belongs to hyperedge, add edge in new graph
            if vertex in hyper_edge_dict[hyperedge]:
                edges.append((vertex, hyperedge))

    # Make adjacency matrix
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N_node, N_node), dtype=np.float32)

    # Makes adj symmetric (I don't understand this part)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + 2.0 * sp.eye(adj.shape[0]))

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # Compute Pv
    dense_Pv = torch.zeros(N_node, N_vertex)

    # Identity projection for existing vertices
    for i in range(N_vertex):
        dense_Pv[i, i] = 1.

    # Projection for added nodes
    for row in range(N_vertex, N_node):
        vertices = hyper_edge_dict.get(row)
        divisor = len(vertices)
        if divisor > 0:
            for column in vertices:
                dense_Pv[row, column] = 1. / divisor

    # Convert to sparse matrix
    Pv = sp.coo_matrix(dense_Pv)

    # Compute PvT
    dense_PvT = torch.zeros(N_vertex, N_node)

    # Approach 1: Remove hyperedge nodes
    if method == 1:
        # Directly project original vertices
        for i in range(N_vertex):
            dense_PvT[i, i] = 1.

    # Approach 2: Incorporate hyperedge nodes
    elif method == 2:
        # Loop over original vertices
        for row in range(N_vertex):
            # Find hyperedges that this vertex was a member of
            hyper_edge_membership = []
            for hyper_edge in hyper_edge_dict.keys():
                if row in hyper_edge_dict.get(hyper_edge):
                    hyper_edge_membership.append(hyper_edge)
            divisor = len(hyper_edge_membership) + 1 # +1 because self included
            for column in hyper_edge_membership:
                dense_PvT[row, column] = 1. / divisor
            dense_PvT[row, row] = 1. / divisor

    # Convert to sparse matrix
    PvT = sp.coo_matrix(dense_PvT)

    return adj, Pv, PvT

def lawler_expansion(pairs, y):
    # Not implemented yet

    return None

def line_graph(pairs, y):

    N_vertex = len(y)

    # Encode hyper-edges
    uniq_hyperedge = np.unique(pairs[:, 1])
    pairs[:, 1] = list(map({hyperedge: i for i, hyperedge in enumerate(uniq_hyperedge)}.get, pairs[:, 1]))
    N_node = len(uniq_hyperedge)

    # Get number of hyperedges = number of vertices in new graph
    hyper_edge_dict = dict()
    for pair in pairs:
        edge_id = pair[1]
        vertex_id = pair[0]
        if hyper_edge_dict.get(edge_id) == None:
            hyper_edge_dict[edge_id] = {vertex_id}
        else:
            hyper_edge_dict[edge_id].add(vertex_id)

    # Add edges
    edges = []

    # Loop over hyperedges (i.e., nodes in new graph)
    for idx_1 in range(N_node):
        for idx_2 in range(idx_1 + 1, N_node):
        #for idx_2 in range(N_node): # change
            #if idx_1 == idx_2: # change
            #    continue
            # Get vertices in each hyperedge
            vertices_1 = hyper_edge_dict.get(idx_1)
            vertices_2 = hyper_edge_dict.get(idx_2)
            if vertices_1 != None and vertices_2 != None:
                # Find intersection
                intersection = vertices_1.intersection(vertices_2)
                # If hyperedges intersect, add an edge
                if len(intersection) > 0:
                    edges.append((idx_1, idx_2))

    # Make adjacency matrix
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N_node, N_node), dtype=np.float32)

    # Makes adj symmetric (I don't understand this part)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + 2.0 * sp.eye(adj.shape[0]))

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # Compute Pv projection
    dense_Pv = torch.zeros(N_node, N_vertex)
    for row in range(N_node):
        vertices = hyper_edge_dict.get(row)
        divisor = len(vertices)
        if divisor > 0:
            for column in vertices:
                dense_Pv[row, column] = 1. / divisor

    # Convert to sparse matrix
    Pv = sp.coo_matrix(dense_Pv)

    # Compute PvT projection
    dense_PvT = torch.zeros(N_vertex, N_node)

    # Loop over old vertices
    for row in range(N_vertex):
        # Find hyperedges that this vertex was a member of
        hyper_edge_membership = []
        for hyper_edge in range(N_node):
            if row in hyper_edge_dict.get(hyper_edge):
                hyper_edge_membership.append(hyper_edge)
        divisor = len(hyper_edge_membership)
        if divisor > 0:
            for column in hyper_edge_membership:
                dense_PvT[row, column] = 1. / divisor

    # Convert to sparse matrix
    PvT = sp.coo_matrix(dense_PvT)

    return adj, Pv, PvT


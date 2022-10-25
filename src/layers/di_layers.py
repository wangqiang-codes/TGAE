# -*- coding: utf-8 -*-
'''
@Time    : 2022/5/6 11:21
@Author  : Wang Qiang
@FileName: di_layers.py
'''

""" Directed Attention layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.nn.conv import GATConv, GATv2Conv

################################################################################
# Activation function
################################################################################
class ActivationFunc(Module):
    """ActivationFunc to compute edge probabilities."""

    def __init__(self, epsilon, t):
        super(ActivationFunc, self).__init__()
        self.epsilon = epsilon
        self.t = t

    def forward(self, d):
        probs = 1. / (torch.exp((self.epsilon - d) / self.t) + 1.0)
        return probs

################################################################################
# DECODER for UNDDIRECTED models
################################################################################
class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


################################################################################
# DECODER for DIRECTED models
################################################################################
class DirectedInnerProductDecoder(torch.nn.Module):
    def __init__(self, epsilon=None, t=None):
        super(DirectedInnerProductDecoder, self).__init__()
        self.act = ActivationFunc(epsilon, t)

    def forward(self, s, t, edge_index, act=None):
        value = (s[edge_index[0]] * t[edge_index[1]]).sum(dim=1)
        if act == 'sigmoid':
            return torch.sigmoid(value)
        elif act == 'relu':
            return torch.relu(value)
        elif act == 'parameterized_sigmoid':
            return self.act(value)
        else:
            return value

    def forward_all(self, s, t, act=None):
        adj = torch.matmul(s, t.t())
        if act == 'sigmoid':
            return torch.sigmoid(adj)
        elif act == 'relu':
            return torch.relu(adj)
        elif act == 'parameterized_sigmoid':
            return self.act(adj)
        else:
            return adj

################################################################################
# DIRECTED model layers: BASIC version
################################################################################
class DirectedGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, activation, alpha, nheads, concat, self_loops, version='GATv1', edge_dim=None, beta=0.0, adaptive=None):
        """Directed torch-geometric version of GAT.
        (torch-geometric has two versions of GAT
        version 1: https://arxiv.org/abs/1710.10903
        version 2: https://arxiv.org/abs/2105.14491)"""
        super(DirectedGNNLayer, self).__init__()
        self.dropout = dropout
        self.act = activation
        self.output_dim = output_dim
        self.version = version
        if version == 'DiGATv1':
            self.source_encoder = GATConv(input_dim, output_dim, heads=nheads, concat=concat, negative_slope=alpha,
                                             dropout=dropout, add_self_loops=self_loops, edge_dim=edge_dim)
            self.target_encoder = GATConv(input_dim, output_dim, heads=nheads, concat=concat, negative_slope=alpha,
                                             dropout=dropout, add_self_loops=self_loops, edge_dim=edge_dim)
        else:
            self.source_encoder = GATv2Conv(input_dim, output_dim, heads=nheads, concat=concat, negative_slope=alpha,
                                             dropout=dropout, add_self_loops=self_loops, edge_dim=edge_dim)
            self.target_encoder = GATv2Conv(input_dim, output_dim, heads=nheads, concat=concat, negative_slope=alpha,
                                             dropout=dropout, add_self_loops=self_loops, edge_dim=edge_dim)

    def forward(self, input):
        s, t, edges, edge_weight = input
        s_h = self.source_encoder((s, t), edges, edge_weight)
        t_h = self.target_encoder((t, s), torch.flip(edges, [0]), edge_weight)
        assert not torch.isnan(s_h).any()
        assert not torch.isnan(t_h).any()
        return (self.act(s_h), self.act(t_h), edges, edge_weight)

def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    acts += [lambda x: x]
    dims = [args.feat_dim] + ([args.dim] * args.num_layers)
    return dims, acts


class DirectedGNNEncoder(nn.Module):
    """
    Directed Graph Neural Networks.
    """

    def __init__(self, args):  # args: num_layers, act, feat_dim, dim, n_heads, alpha, dropout
        super(DirectedGNNEncoder, self).__init__()
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            if args.concat == True:
                assert dims[i + 1] % args.n_heads == 0
                out_dim = dims[i + 1] // args.n_heads
            else:
                out_dim = dims[i + 1]
            gat_layers.append(
                DirectedGNNLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, args.concat,
                                            args.self_loops, version=args.encoder, edge_dim=args.weight_dim))
        self.layers = nn.Sequential(*gat_layers)

    def forward(self, s, t, edge_index, edge_weight=None):
        input = (s, t, edge_index, edge_weight)
        s_out, t_out, _, _ = self.layers.forward(input)
        return s_out, t_out
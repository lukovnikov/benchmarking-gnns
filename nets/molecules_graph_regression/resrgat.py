import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.resrgat_layer import ResRGATCell

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.mlp_readout_layer import MLPReadout


class ResRGATNet(torch.nn.Module):
    def __init__(self, params):
        super(ResRGATNet, self).__init__()
        num_atom_type = params['num_atom_type']
        num_bond_type = params['num_bond_type']
        # embdim = params['embedding_dim']
        self.hdim = params["hidden_dim"]
        self.norel = params["norel"] if "norel" in params else False
        self.edge_feat = params["edge_feat"]
        self.pos_enc = params["pos_enc"]
        self.readout = params["readout"]
        numheads = params["n_heads"]
        dropout = params["dropout"]
        dropoutemb = params["in_feat_dropout"]
        dropout_red = params["dropout_red"] if "dropout_red" in params else 0.
        dropout_attn = params["dropout_attn"] if "dropout_attn" in params else 0.
        # dropout_act = params["dropout_act"] if "dropout_act" in params else 0.
        rdim = params["rdim"] if "rdim" in params else self.hdim
        usevallin = params["usevallin"] if "usevallin" in params else False
        cat_rel = params["cat_rel"] if "cat_rel" in params else True
        cat_tgt = params["cat_tgt"] if "cat_tgt" in params else False
        use_gate = params["use_gate"] if "use_gate" in params else False
        skipatt = params["skipatt"] if "skipatt" in params else False
        self.numlayers = params["numlayers"] if "numlayers" in params else 1
        # self.numrepsperlayer = params["numrepsperlayer"] if "numrepsperlayer" in params else 1
        self.numrepsperlayer = params["L"]
        self.device = params['device']

        self.embedding_h = nn.Embedding(num_atom_type, self.hdim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, self.hdim)
        else:
            self.embedding_e = nn.Linear(1, self.hdim)

        self.in_feat_dropout = nn.Dropout(dropoutemb)

        self.layers = torch.nn.ModuleList([
            ResRGATCell(self.hdim, numrels=1, numheads=numheads,
                      dropout=0., dropout_red=dropout_red, dropout_attn=dropout_attn,
                      dropout_act=dropout,
                      rdim=rdim, usevallin=usevallin, norel=self.norel,
                      cat_rel=cat_rel, cat_tgt=cat_tgt, use_gate=use_gate,
                      skipatt=skipatt)
            for _ in range(self.numlayers)
        ])
        self.dropout = torch.nn.Dropout(dropout)
        self.MLP_layer = MLPReadout(self.hdim, 1)   # 1 out dim since regression problem

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, h, e, h_pos_enc=None):
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc
        if not self.edge_feat:  # edge feature set to 1
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.embedding_e(e)

        g.ndata["h"] = h
        g.edata["emb"] = e

        # convnets
        for layer in self.layers:
            for _ in range(self.numrepsperlayer):
                g = layer(g)

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        hg = self.dropout(hg)

        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss


# class GatedGCNNet(nn.Module):
#     def __init__(self, net_params):
#         super().__init__()
#         num_atom_type = net_params['num_atom_type']
#         num_bond_type = net_params['num_bond_type']
#         hidden_dim = net_params['hidden_dim']
#         out_dim = net_params['out_dim']
#         in_feat_dropout = net_params['in_feat_dropout']
#         dropout = net_params['dropout']
#         n_layers = net_params['L']
#         self.readout = net_params['readout']
#         self.batch_norm = net_params['batch_norm']
#         self.residual = net_params['residual']
#         self.edge_feat = net_params['edge_feat']
#         self.device = net_params['device']
#         self.pos_enc = net_params['pos_enc']
#         if self.pos_enc:
#             pos_enc_dim = net_params['pos_enc_dim']
#             self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
#
#         self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
#
#         if self.edge_feat:
#             self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
#         else:
#             self.embedding_e = nn.Linear(1, hidden_dim)
#
#         self.in_feat_dropout = nn.Dropout(in_feat_dropout)
#
#         self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
#                                                     self.batch_norm, self.residual) for _ in range(n_layers-1) ])
#         self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
#         self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem
#
#     def forward(self, g, h, e, h_pos_enc=None):
#
#         # input embedding
#         h = self.embedding_h(h)
#         h = self.in_feat_dropout(h)
#         if self.pos_enc:
#             h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
#             h = h + h_pos_enc
#         if not self.edge_feat: # edge feature set to 1
#             e = torch.ones(e.size(0),1).to(self.device)
#         e = self.embedding_e(e)
#
#         # convnets
#         for conv in self.layers:
#             h, e = conv(g, h, e)
#         g.ndata['h'] = h
#
#         if self.readout == "sum":
#             hg = dgl.sum_nodes(g, 'h')
#         elif self.readout == "max":
#             hg = dgl.max_nodes(g, 'h')
#         elif self.readout == "mean":
#             hg = dgl.mean_nodes(g, 'h')
#         else:
#             hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
#
#         return self.MLP_layer(hg)
#
#     def loss(self, scores, targets):
#         # loss = nn.MSELoss()(scores,targets)
#         loss = nn.L1Loss()(scores, targets)
#         return loss

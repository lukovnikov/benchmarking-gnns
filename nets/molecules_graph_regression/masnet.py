import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import networkx as nx

from layers.mas import MASNN, Readout, MaskedReadout
from layers.resrgat_layer import ResRGATCell

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.mlp_readout_layer import MLPReadout
from dgl.nn.pytorch.glob import Set2Set


class MASNet(torch.nn.Module):
    def __init__(self, params):
        super(MASNet, self).__init__()
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
        dropout_red = params["dropout_red"] if "dropout_red" in params else 0.
        dropout_attn = params["dropout_attn"] if "dropout_attn" in params else 0.
        # dropout_act = params["dropout_act"] if "dropout_act" in params else 0.
        rdim = params["rdim"] if "rdim" in params else self.hdim
        self.numlayers = params["numlayers"] if "numlayers" in params else 1
        self.numsublayers = params["numsublayers"] if "numsublayers" in params else 0
        self.numechos = params["numechos"] if "numechos" in params else 0
        # self.numrepsperlayer = params["numrepsperlayer"] if "numrepsperlayer" in params else 1
        self.numrepspergnnlayer = params["L"]
        self.device = params['device']

        use_sgru = params["use_sgru"] if "use_sgru" in params else False

        self.embedding_h = nn.Embedding(num_atom_type, self.hdim)

        if self.edge_feat:
            # self.embedding_e = nn.Embedding(num_bond_type+1, self.hdim)
            # self.self_edge_id = num_bond_type
            pass
        else:
            raise Exception("not implemented")
            # self.embedding_e = nn.Linear(1, self.hdim)
        self.self_edge_id = num_bond_type

        self.in_feat_dropout = nn.Dropout(dropout)

        self.net = MASNN(self.hdim, numlayers=self.numlayers, numgnnperlayer=self.numsublayers, numrepsperlayer=1,
                         numrepspergnnlayer=self.numrepspergnnlayer,
                         numechos=self.numechos, numrels=num_bond_type+1, numheads=numheads,
                         dropout=dropout, use_sgru=use_sgru)

        self.dropout = torch.nn.Dropout(dropout)
        self.MLP_layer = MLPReadout(self.hdim, 1)   # 1 out dim since regression problem

        if self.readout == "set2set":
            self.set2set = Set2Set(self.hdim, n_iters=10, n_layers=1)
        elif "mas" in self.readout:
            self.readoutm = MaskedReadout(mode=self.readout, maskattr="iscentr")
        else:
            self.readoutm = Readout(mode=self.readout)

    def init_node_states(self, g, batsize, device):
        self.net.init_node_states(g, batsize, device)

    def forward(self, g, h, e, h_pos_enc=None):
        g = g.local_var()
        gs = dgl.unbatch(g)
        _gs = []
        for gse in gs:
            # extra_e = torch.ones(gse.number_of_nodes(), device=e.device, dtype=e.dtype) * self.self_edge_id
            # e = torch.cat([e, extra_e], 0)
            nodeids = torch.arange(gse.number_of_nodes(), dtype=h.dtype, device=h.device)
            gse.add_edges(nodeids, nodeids, data={"feat": torch.ones_like(nodeids, dtype=torch.long) * self.self_edge_id})
            _gs.append(gse)
            # assert torch.all(g.edata["feat"] == e)
        g = dgl.batch(_gs)
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc
        if not self.edge_feat:  # edge feature set to 1
            raise Exception("not implemented yet")
        #     e = torch.ones(e.size(0), 1).to(self.device)
        # e = self.embedding_e(e)

        g.ndata["h"] = h
        g.edata["id"] = g.edata["feat"]

        # convnets
        self.net.init_node_states(g, h.size(0), device=h.device)
        self.net.reset_dropout()
        g = self.net(g)

        g_out = self.readoutm(g, "h")
        hg = self.dropout(g_out)

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

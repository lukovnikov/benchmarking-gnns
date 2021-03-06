import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import networkx as nx

from layers.resrgat_layer import ResRGATCell

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.mlp_readout_layer import MLPReadout
from dgl.nn.pytorch.glob import Set2Set
#
# class Set2Set(torch.nn.Module):
#     r"""
#     Set2Set global pooling operator from the `"Order Matters: Sequence to sequence for sets"
#     <https://arxiv.org/abs/1511.06391>`_ paper. This pooling layer performs the following operation
#     .. math::
#         \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})
#         \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)
#         \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i
#         \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,
#     where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
#     the dimensionality as the input.
#     Arguments
#     ---------
#         input_dim: int
#             Size of each input sample.
#         hidden_dim: int, optional
#             the dim of set representation which corresponds to the input dim of the LSTM in Set2Set.
#             This is typically the sum of the input dim and the lstm output dim. If not provided, it will be set to :obj:`input_dim*2`
#         steps: int, optional
#             Number of iterations :math:`T`. If not provided, the number of nodes will be used.
#         num_layers : int, optional
#             Number of recurrent layers (e.g., :obj:`num_layers=2` would mean stacking two LSTMs together)
#             (Default, value = 1)
#     """
#
#     def __init__(self, nin, nhid=None, steps=None, num_layers=1, activation=None, device='cpu'):
#         super(Set2Set, self).__init__()
#         self.steps = steps
#         self.nin = nin
#         self.nhid = nin * 2 if nhid is None else nhid
#         if self.nhid <= self.nin:
#             raise ValueError('Set2Set hidden_dim should be larger than input_dim')
#         # the hidden is a concatenation of weighted sum of embedding and LSTM output
#         self.lstm_output_dim = self.nhid - self.nin
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(self.nhid, self.nin, num_layers=num_layers, batch_first=True).to(device)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         r"""
#         Applies the pooling on input tensor x
#         Arguments
#         ----------
#             x: torch.FloatTensor
#                 Input tensor of size (B, N, D)
#         Returns
#         -------
#             x: `torch.FloatTensor`
#                 Tensor resulting from the  set2set pooling operation.
#         """
#
#         batch_size = x.shape[0]
#         n = self.steps or x.shape[1]
#
#         h = (x.new_zeros((self.num_layers, batch_size, self.nin)),
#              x.new_zeros((self.num_layers, batch_size, self.nin)))
#
#         q_star = x.new_zeros(batch_size, 1, self.nhid)
#
#         for i in range(n):
#             # q: batch_size x 1 x input_dim
#             q, h = self.lstm(q_star, h)
#             # e: batch_size x n x 1
#             e = torch.matmul(x, torch.transpose(q, 1, 2))
#             a = self.softmax(e)
#             r = torch.sum(a * x, dim=1, keepdim=True)
#             q_star = torch.cat([q, r], dim=-1)
#
#         return torch.squeeze(q_star, dim=1)


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

        use_sgru = params["use_sgru"] if "use_sgru" in params else False
        use_logdegree = params["use_logdegree"] if "use_logdegree" in params else False

        if self.numrepsperlayer > 8:
            self.numlayers = int(self.numrepsperlayer / 2)
            self.numrepsperlayer = 2
        else:
            self.numlayers = self.numrepsperlayer
            self.numrepsperlayer = 1

        self.embedding_h = nn.Embedding(num_atom_type, self.hdim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type+1, self.hdim)
            self.self_edge_id = num_bond_type
        else:
            self.embedding_e = nn.Linear(1, self.hdim)
            self.self_edge_id = num_bond_type

        self.in_feat_dropout = nn.Dropout(dropoutemb)

        self.layers = torch.nn.ModuleList([
            ResRGATCell(self.hdim, numrels=10, numheads=numheads,
                      dropout=dropout, dropout_red=dropout_red, dropout_attn=dropout_attn,
                      dropout_act=0.,
                      rdim=rdim, usevallin=usevallin, norel=self.norel,
                      cat_rel=cat_rel, cat_tgt=cat_tgt, use_gate=use_gate,
                      skipatt=skipatt, use_sgru=use_sgru, use_logdegree=use_logdegree)
            for _ in range(self.numlayers)
        ])
        self.dropout = torch.nn.Dropout(dropout)
        self.MLP_layer = MLPReadout(self.hdim, 1)   # 1 out dim since regression problem

        if self.readout == "set2set":
            self.set2set = Set2Set(self.hdim, n_iters=10, n_layers=1)

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, h, e, h_pos_enc=None):
        g = g.local_var()
        extra_e = torch.ones(h.size(0), device=e.device, dtype=e.dtype) * self.self_edge_id
        e = torch.cat([e, extra_e], 0)
        nodeids = torch.arange(h.size(0), dtype=h.dtype, device=h.device)
        g.add_edges(nodeids, nodeids, data={"feat": torch.ones_like(nodeids, dtype=torch.long) * self.self_edge_id})
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
        g.edata["id"] = g.edata["feat"]

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
        elif self.readout == "set2set":
            hg = self.set_to_set(g)
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

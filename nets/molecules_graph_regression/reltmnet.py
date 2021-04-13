import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import networkx as nx

from layers.lrtm_layer import LRTMCell
from layers.reltm_layer import RelTMCell

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.mlp_readout_layer import MLPReadout
from dgl.nn.pytorch.glob import Set2Set


class RelTM(torch.nn.Module):
    def __init__(self, params):
        super(RelTM, self).__init__()
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

        self.embedding_h = nn.Embedding(num_atom_type, self.hdim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type+1, self.hdim)
            self.self_edge_id = num_bond_type
        else:
            self.embedding_e = nn.Linear(1, self.hdim)
            self.self_edge_id = num_bond_type

        self.in_feat_dropout = nn.Dropout(dropoutemb)

        self.layers = torch.nn.ModuleList([
            RelTMCell(self.hdim, numrels=10, numheads=numheads, dropout=dropout, norel=self.norel, use_gate=use_gate)
            for _ in range(self.numlayers)
        ])
        self.numrepsperlayer = self.numrepsperlayer
        self.dropout = torch.nn.Dropout(dropout)
        self.MLP_layer = MLPReadout(self.hdim, 1)   # 1 out dim since regression problem

        if self.readout == "set2set":
            self.set2set = Set2Set(self.hdim, n_iters=10, n_layers=1)

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, h, e, h_pos_enc=None):
        # g = g.local_var()
        # extra_e = torch.ones(h.size(0), device=e.device, dtype=e.dtype) * self.self_edge_id
        # e = torch.cat([e, extra_e], 0)
        # nodeids = torch.arange(h.size(0), dtype=h.dtype, device=h.device)
        # g.add_edges(nodeids, nodeids, data={"feat": torch.ones_like(nodeids, dtype=torch.long) * self.self_edge_id})
        # # input embedding
        # h = self.embedding_h(h)
        # h = self.in_feat_dropout(h)
        # if self.pos_enc:
        #     h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
        #     h = h + h_pos_enc
        # if not self.edge_feat:  # edge feature set to 1
        #     e = torch.ones(e.size(0), 1).to(self.device)
        # e = self.embedding_e(e)
        #
        # g.ndata["h"] = h
        # g.edata["emb"] = e


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


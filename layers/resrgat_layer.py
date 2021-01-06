import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""


class MultiHeadAttention(torch.nn.Module):
    GUMBEL_TEMP = 1.
    def __init__(self, querydim, keydim=None, valdim=None, hdim=None,
                 use_layernorm=False, dropout=0., attn_dropout=0., numheads=1, usevallin=True,
                 gumbelmix=0., **kw):
        super(MultiHeadAttention, self).__init__(**kw)
        self.querydim = querydim
        self.hdim = querydim if hdim is None else hdim
        self.valdim = querydim if valdim is None else valdim
        self.keydim = querydim if keydim is None else keydim
        self.usevallin = usevallin
        self.numheads = numheads
        assert(self.hdim // self.numheads == self.hdim / self.numheads)

        self.query_lin = torch.nn.Linear(self.querydim, self.hdim, bias=False)
        self.key_lin = torch.nn.Linear(self.keydim, self.hdim, bias=False)
        self.value_lin = torch.nn.Linear(self.valdim, self.hdim, bias=False) if usevallin is True else None

        self.dropout = torch.nn.Dropout(dropout)
        self.attn_dropout = torch.nn.Dropout(attn_dropout)

        if use_layernorm:
            self.ln_query = torch.nn.LayerNorm(querydim)
            self.ln_key = torch.nn.LayerNorm(keydim)
            if self.value_lin is not None:
                self.ln_value = torch.nn.LayerNorm(valdim)
        else:
            self.ln_query = None
            self.ln_key = None
            self.ln_value = None

        self.gumbelmix = gumbelmix

    def forward(self, query, key, val):
        if self.ln_query is not None:
            query = self.ln_query(query)
        if self.ln_key is not None:
            key = self.ln_key(key)
        if self.ln_value is not None:
            val = self.ln_value(val)
        queries = self.query_lin(query)
        context = self.key_lin(key)    # bsd
        queries = queries.view(queries.size(0), self.numheads, -1)  # bhl
        context = context.view(context.size(0), context.size(1), self.numheads, -1).transpose(1, 2)     # bhsl
        weights = torch.einsum("bhd,bhsd->bhs", queries, context) / math.sqrt(context.size(-1))
        alphas = torch.softmax(weights, -1)      # bhs

        alphas = self.attn_dropout(alphas)
        values = val
        if self.value_lin is not None:
            values = self.value_lin(values)
        values = values.view(values.size(0), values.size(1), self.numheads, -1).transpose(1, 2)
        red = torch.einsum("bhs,bhsd->bhd", alphas, values)
        red = red.view(red.size(0), -1)
        return red


class ResRGATCell(torch.nn.Module):
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0., dropout_attn=0., dropout_act=0.,
                 dropout_red=0., rdim=None, usevallin=False, norel=False,
                 cat_rel=True, cat_tgt=False, use_gate=False,
                 skipatt=False, **kw):
        super(ResRGATCell, self).__init__(**kw)
        self.cat_rel, self.cat_tgt = cat_rel, cat_tgt
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        indim = self.hdim + (self.hdim if self.cat_tgt else 0) + (self.rdim if True else 0)
        self.zdim = hdim
        self.numrels = numrels
        self.norel = norel
        self.skipatt = skipatt
        self.use_gate = use_gate

        if norel:
            self.cat_rel = False
        else:
            self.activation = torch.nn.CELU()

            self.linA = torch.nn.Linear(indim, self.zdim)
            self.linB = torch.nn.Linear(self.zdim, self.hdim)

            self.ln = torch.nn.LayerNorm(indim)
            self.ln2 = torch.nn.LayerNorm(self.hdim)
            self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
            init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))

        self.linGate = None
        if self.use_gate:
            self.linGate = torch.nn.Linear(self.zdim, self.hdim)
            self.linGate.bias.data.fill_(3.)

        self.attention = MultiHeadAttention(self.hdim, (self.hdim + self.rdim) if self.cat_rel else self.hdim,
                                            self.hdim, self.hdim, numheads=numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=dropout_attn, usevallin=usevallin)

        self.dropout = torch.nn.Dropout(dropout)
        self.dropout_red = torch.nn.Dropout(dropout_red)
        self.dropout_act = torch.nn.Dropout(dropout_act)

        # ablations
        self.usevallin = usevallin

        self.ln_att = torch.nn.LayerNorm(hdim)

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        if self.norel:
            relvecs = torch.zeros(hs.size(0), self.rdim)
        elif "emb" in edges.data:             # if "emb" in edata, then use those
            relvecs = edges.data["emb"]
        else:
            relvecs = self.relvectors[edges.data["id"]]
        inps = [hs, relvecs]
        if self.cat_tgt:    # False by default
            inps.append(edges.dst["h"])

        # residual update
        x = torch.cat(inps, -1)
        x = self.dropout_act(x)
        x = self.ln(x)
        x = self.linA(x)
        x = self.activation(x)
        # x = self.dropout_act(x)
        _x = self.linB(x)
        # _x = self.ln2(_x)
        _x = self.dropout(_x)
        if self.use_gate:
            g = self.linGate(x)
            g = torch.sigmoid(g)
            hs = hs * g + _x * (1 - g)
        else:
            hs = hs + _x
        # if not self.skipatt:
        #     hs = self.ln2(hs)

        if self.cat_rel:  # True by default
            msg = torch.cat([hs, relvecs], -1)
        else:
            msg = hs

        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
        red = self.attention(queries, keys, values)
        if self.skipatt:
            red = red + nodes.data["h"]
            red = self.ln_att(red)
        return {"red": red}

    def apply_node_func(self, nodes):
        h = nodes.data["red"]
        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g
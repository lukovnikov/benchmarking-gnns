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


GATE_BIAS = 0
class DGRUCell(torch.nn.Module):
    zoneout_frac = .7
    def __init__(self, dim, bias=True, dropout=0., gate_bias=GATE_BIAS, use_layernorm=True, **kw):
        super(DGRUCell, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.gateW = torch.nn.Linear(dim * 2, dim * 5, bias=bias)
        self.gateU = torch.nn.Linear(dim * 2, dim, bias=bias)
        self.sm = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("dropout_mask", torch.ones(self.dim * 2))
        self.gate_bias = gate_bias

        self.ln = None
        self.ln2 = None
        if use_layernorm:
            self.ln = torch.nn.LayerNorm(dim * 2)
            self.ln2 = torch.nn.LayerNorm(dim * 2)

    def reset_dropout(self):
        pass

    def forward(self, x, h):
        inp = torch.cat([x, h], 1)
        if self.ln is not None:
            inp = self.ln(inp)
        inp = self.dropout(inp)
        inp = inp * self.dropout_mask[None, :]
        gates = self.gateW(inp)
        gates = list(gates.chunk(5, 1))
        rx = torch.sigmoid(gates[0])
        rh = torch.sigmoid(gates[1])
        z_gates = gates[2:5]
        z_gates[2] = z_gates[2] - self.gate_bias
        z = torch.softmax(torch.stack(z_gates, -1), -1)
        inp = torch.cat([x * rx, h * rh], 1)
        if self.ln2 is not None:
            inp = self.ln2(inp)
        inp = self.dropout(inp)
        inp = inp * self.dropout_mask[None, :]
        u = self.gateU(inp)
        u = torch.tanh(u)
        h_new = torch.stack([x, h, u], 2) * z
        h_new = h_new.sum(-1)
        return h_new


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


class FFN(torch.nn.Module):
    GATE_BIAS = -2.5
    def __init__(self, hdim, indim=None, zdim=None, use_gate=False, dropout=0., act_dropout=0., **kw):
        super(FFN, self).__init__(**kw)
        self.hdim = hdim
        self.indim = indim if indim is not None else self.hdim
        self.zdim = zdim if zdim is not None else self.hdim
        self.use_gate = use_gate

        self.ln_inp = torch.nn.LayerNorm(self.indim)
        # self.ln_att = torch.nn.LayerNorm(self.hdim)
        # self.ln_final = torch.nn.LayerNorm(self.hdim)

        self.dropout = torch.nn.Dropout(dropout)
        self.act_dropout = torch.nn.Dropout(act_dropout)

        self.act_fn = torch.nn.CELU()

        self.fc1 = torch.nn.Linear(self.indim, self.zdim)
        if self.use_gate:
            self.fc2 = torch.nn.Linear(self.zdim, self.hdim * 2)
            self.fc2.bias.data[self.hdim:] = self.GATE_BIAS
            # self.fcg = torch.nn.Linear(self.zdim, self.hdim)
            # self.fcg.bias.data.fill_(-2.5)
        else:
            self.fc2 = torch.nn.Linear(self.zdim, self.hdim)

    def forward(self, x):
        z = x
        z = self.ln_inp(z)
        z = self.fc1(z)
        z = self.act_fn(z)
        z = self.act_dropout(z)
        z = self.fc2(z)
        if self.use_gate:
            x, r = torch.chunk(z, 2, -1)
            x = torch.sigmoid(r) * self.dropout(x)
        else:
            x = self.dropout(z)
        return x


class LRTMCell(torch.nn.Module):   # Light Relational Transformer
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0., attn_dropout=0., act_dropout=0.,
                 norel=False, use_gate=True, use_sgru=False, skipatt=False, **kw):
        super(LRTMCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim
        self.zdim = self.hdim
        self.numrels = numrels
        self.norel = norel
        self.skipatt = skipatt
        self.use_gate = use_gate

        self.attention = MultiHeadAttention(self.hdim, self.hdim,
                                            self.hdim, self.hdim, numheads=numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=attn_dropout,
                                            usevallin=False)
        # self.attention_lin = torch.nn.Linear(self.hdim, self.hdim)

        self.use_sgru = use_sgru
        if self.use_sgru:
            self.sgru = DGRUCell(self.hdim, dropout=dropout)
        else:
            self.update_fn = FFN(self.hdim, dropout=dropout, act_dropout=act_dropout, use_gate=self.use_gate)

        if not self.norel:
            # self.msg_ln = torch.nn.LayerNorm(self.hdim, elementwise_affine=False)
            self.relvectors = torch.nn.Embedding(self.numrels, self.hdim*3)
            torch.nn.init.kaiming_uniform_(self.relvectors.weight.data[:, :self.hdim], a=math.sqrt(5))
            torch.nn.init.ones_(self.relvectors.weight.data[:, self.hdim:self.hdim*2])
            torch.nn.init.zeros_(self.relvectors.weight.data[:, self.hdim*2:])

            # message function network from ResRGAT
            self.msg_fn = FFN(indim=self.hdim + self.rdim, hdim=self.hdim, use_gate=self.use_gate, dropout=dropout, act_dropout=act_dropout)

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
            relvecs = self.relvectors(edges.data["id"])
        relvecs = relvecs[:, :self.hdim]
        inps = [hs, relvecs]

        # residual update
        x = torch.cat(inps, -1)
        _x = self.msg_fn(x)
        hs = hs + _x
        msg = hs

        return {"msg": msg, "hs": hs}

    def simple_message_func(self, edges):
        hs = edges.src["h"]
        if self.norel:
            return {"msg": hs, "hs": hs}
        if self.norel:
            relvecs = torch.ones(hs.size(0), self.rdim)
            relvecs2 = torch.zeros(hs.size(0), self.rdim)
            relvecs_add = torch.ones(hs.size(0), self.rdim)
        else:
            if "emb" in edges.data:  # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors(edges.data["id"])
            relvecs_add, relvecs, relvecs2 = torch.chunk(relvecs, 3, -1)

        _hs = hs
        hs = hs + relvecs_add
        hs = self.msg_ln(hs)
        hs = hs * relvecs + relvecs2
        # hs = hs + relvecs2
        hs = torch.nn.functional.leaky_relu(hs, 0.25)

        hs = _hs + hs       # residual

        msg = hs
        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
        red = self.attention(queries, keys, values)
        return {"red": red}

    def apply_node_func(self, nodes):
        if self.use_sgru:
            h = self.sgru(nodes.data["h"], nodes.data["red"])
        else:
            h = nodes.data["h"]
            summ = nodes.data["red"]

            if self.skipatt:
                summ = summ + h
                summ = self.ln_att(summ)

            z = summ
            x = self.update_fn(z)
            h = h + x
            # h = summ + x

        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g

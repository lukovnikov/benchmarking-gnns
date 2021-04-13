import math
import random
import numpy as np

import dgl
import torch

from layers.lrtm_layer import LRTMCell
from nets.molecules_graph_regression.lrtmnet import LRTM


class DeepRGCN(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numrels=1, dropout=0.,
                 residual=True, **kw):
        super(DeepRGCN, self).__init__(**kw)
        self.hdim = hdim
        self.layers = torch.nn.ModuleList([
            DeepGCNCell(hdim, numrels=numrels, dropout=dropout, **kw)
            for _ in range(numlayers)
        ])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hdim) for _ in range(numlayers)])
        self.dropout = torch.nn.Dropout(dropout)
        self.residual = residual
        self.numlayers = numlayers

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, step=None):
        assert(step is None)
        g = self.layers[0](g)
        for layernr in range(1, self.numlayers):
            h = g.ndata["h"]
            _h = self.norms[layernr-1](h)
            _h = torch.relu(_h)
            _h = self.dropout(_h)
            g.ndata["h"] = _h

            g = self.layers[layernr](g)

            if self.residual:
                _h = g.ndata["h"]
                _h = h + _h
                g.ndata["h"] = _h

        h = g.ndata["h"]
        h = self.norms[-1](h)
        h = self.dropout(h)
        g.ndata["h"] = h
            # _step = step/self.numrepsperlayer
            # _step = math.floor(_step)
            # _step = min(_step, len(self.layers) - 1)
            # layer = self.layers[_step]
            # norm = self.norms[_step]
            #
            # g = layer(g, step=None)
            # g = norm
            # g = layer(g, step=None)
        return g


class DeepGCNCell(torch.nn.Module):
    def __init__(self, dim, dropout=0., numrels=5, residual=True, **kw):
        super(DeepGCNCell, self).__init__(**kw)
        self.lin = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.residual = residual
        self.ln = torch.nn.LayerNorm(dim)
        self.hdim = dim

        self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.hdim))
        torch.nn.init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        if "emb" in edges.data:
            relvecs = edges.data["emb"]
        else:
            relvecs = self.relvectors[edges.data["id"]]
        msg = hs + relvecs
        # msg = self.ln(msg)
        msg = torch.relu(msg)
        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        msg, hs = nodes.mailbox["msg"], nodes.mailbox["hs"]
        degree = hs.size(1)
        red = msg.mean(1)
        # deg_inv_sqrt = degree ** -0.5
        # if deg_inv_sqrt == float('inf'):
        #     deg_inv_sqrt = 0
        # red = red * deg_inv_sqrt
        return {"red": red, "degree": torch.ones_like(red[:, 0])*degree}

    def apply_node_func(self, nodes):
        h = nodes.data["h"]
        red = nodes.data["red"]
        # degree = nodes.data["degree"]
        # # _h =
        # deg_inv_sqrt = degree.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # red = red * deg_inv_sqrt[:, None]
        # _h = h + red
        red = self.lin(red)
        return {"h": red}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g



class SimpleRGAT(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numrepsperlayer=1, numrels=1, numheads=4, dropout=0.,
                 rdim=None, residual=True, **kw):
        super(SimpleRGAT, self).__init__(**kw)
        self.hdim = hdim
        self.layers = torch.nn.ModuleList([
            SimpleRGATCell(hdim, numrels=numrels, numheads=numheads, dropout=dropout,
                      rdim=rdim, residual=residual, **kw)
            for _ in range(numlayers)
        ])
        self.numrepsperlayer = numrepsperlayer
        self.residual = residual

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, step=None):
        if step is None:
            for layer in self.layers:
                for _ in range(self.numrepsperlayer):
                    g = layer(g)
        else:
            # assert(step is not None)
            _step = step/self.numrepsperlayer
            _step = math.floor(_step)
            _step = min(_step, len(self.layers) - 1)
            layer = self.layers[_step]
            g = layer(g, step=None)
        return g


class SimpleRGATCell(torch.nn.Module):   # same as SGGNN but without all the ablations
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0., attn_dropout=0., act_dropout=0.,
                 rdim=None, residual=True, **kw):
        super(SimpleRGATCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        self.zdim = self.hdim * 2
        self.numrels = numrels

        self.attention = MultiHeadAttention(self.hdim, self.hdim,
                                            self.hdim, self.hdim, numheads=numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=attn_dropout, usevallin=True)
        self.attention_lin = torch.nn.Linear(self.hdim, self.hdim)
        self.ln_att = torch.nn.LayerNorm(self.hdim)
        self.ln_final = torch.nn.LayerNorm(self.hdim)

        self.dropout = torch.nn.Dropout(dropout)
        self.act_dropout = torch.nn.Dropout(act_dropout)

        self.act_fn = torch.nn.CELU()

        self.fc1 = torch.nn.Linear(self.hdim, self.zdim)
        self.fc2 = torch.nn.Linear(self.zdim, self.hdim)

        self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
        torch.nn.init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
        self.lrelu = torch.nn.LeakyReLU(0.25)

        self.residual = residual

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]

        if "emb" in edges.data:             # if "emb" in edata, then use those
            relvecs = edges.data["emb"]
        else:
            relvecs = self.relvectors[edges.data["id"]]
        hs = hs + relvecs
        hs = self.lrelu(hs)
        msg = hs
        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
        red = self.attention(queries, keys, values)
        return {"red": red}

    def apply_node_func(self, nodes):
        h = nodes.data["h"]
        summ = nodes.data["red"]
        x = summ
        x = self.act_fn(x)
        x = self.dropout(x)
        if self.residual:
            h = h + x
        else:
            h = x
        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g


def compute_dags(g:dgl.DGLGraph, include_self=True):
    # TODO: implement this more efficiently
    """
    :param g:       must be a dgl graph with computed centralities
    :return:        two dgl graphs that are dags
    """
    edges = g.edges()
    edges = list(zip(list(edges[0].cpu().numpy()), list(edges[1].cpu().numpy())))
    original_edges = edges
    edges = edges[:]
    edge_to_id = {edge: i for i, edge in enumerate(edges)}
    edges_in_dag1 = set()

    # precompute edges incoming and outgoing per node
    incoming_edges_per_node = {i: [] for i in range(g.number_of_nodes())}
    outgoing_edges_per_node = {i: [] for i in range(g.number_of_nodes())}
    for edge in edges:
        incoming_edges_per_node[edge[1]].append(edge)
        outgoing_edges_per_node[edge[0]].append(edge)

    nodes_in_dag1 = set()
    # pick a center
    centr = g.ndata["iscentr"].nonzero()
    # centr = (g.ndata["centr"] > 0.5).nonzero()
    # print(centr[:, 0])
    centr = random.choice(list(centr[:, 0].cpu().numpy()))

    nodes_in_dag1.add(centr)

    # find next level of nodes
    newedges = edges[:]
    while len(newedges) > 0:
        random.shuffle(newedges)
        nextlevel_nodes = []
        for edge in newedges:
            if edge[1] in nodes_in_dag1:
                nextlevel_nodes.append(edge[0])

        # add next level of nodes and edges
        random.shuffle(nextlevel_nodes)
        dobreak = True
        for node in nextlevel_nodes:
            if node in nodes_in_dag1:
                continue
            outgoing_edges = outgoing_edges_per_node[node][:]
            # random.shuffle(outgoing_edges)
            for edge in outgoing_edges:  # for every edge that goes out of the new node
                if edge[1] in nodes_in_dag1:    # if that edge ends in a node in our dag selected so far
                    edges_in_dag1.add(edge)     # add the edge to the dag
            nodes_in_dag1.add(node)     # add node to the dag
            dobreak = False

        newedges = [edge for edge in newedges if edge not in edges_in_dag1]
        if dobreak:
            break

    edges_in_dag2 = [edge for edge in newedges if edge[0] != edge[1]]
    self_edges = [edge for edge in newedges if edge[0] == edge[1]]

    if include_self:
        edges_in_dag1 = list(edges_in_dag1) + self_edges
        edges_in_dag2 = list(edges_in_dag2) + self_edges

    dag1 = dgl.DGLGraph()
    dag2 = dgl.DGLGraph()
    dag1.add_nodes(g.number_of_nodes(), data=g.ndata)
    if not (max(nodes_in_dag1) == g.number_of_nodes()-1 and len(nodes_in_dag1) == g.number_of_nodes()):
        assert(max(nodes_in_dag1) == g.number_of_nodes()-1 and len(nodes_in_dag1) == g.number_of_nodes())
    dag2.add_nodes(g.number_of_nodes(), data=g.ndata)

    dag1_edgeids = []
    dag2_edgeids = []
    for edgeid, edge in enumerate(original_edges):
        # data = {k:v[edgeid:edgeid+1] for k, v in g.edata.items()}
        if edge in edges_in_dag1:
            dag1_edgeids.append(edgeid)
            # dag1.add_edge(edge[0], edge[1], data=data)
        if edge in edges_in_dag2:
            dag2_edgeids.append(edgeid)
            # dag2.add_edge(edge[0], edge[1], data=data)
    device = g.ndata["h"].device  #list(g.ndata.items())[0][1].device
    dag1_edgeids = torch.tensor(dag1_edgeids).to(device)
    dag2_edgeids = torch.tensor(dag2_edgeids).to(device)

    u, v = g.edges()
    u1, u2 = u[dag1_edgeids], u[dag2_edgeids]
    v1, v2 = v[dag1_edgeids], v[dag2_edgeids]

    d1 = {k: v[dag1_edgeids] for k, v in g.edata.items()}
    d2 = {k: v[dag2_edgeids] for k, v in g.edata.items()}

    dag1.add_edges(u1, v1, data=d1)
    dag2.add_edges(u2, v2, data=d2)

    # plot_graph(dag1)
    # plot_graph(dag2)
    return dag1, dag2


class MultiHeadAttention(torch.nn.Module):
    GUMBEL_TEMP = 1.
    def __init__(self, querydim, keydim=None, valdim=None, hdim=None,
                 use_layernorm=False, dropout=0., attn_dropout=0., numheads=1, usevallin=True,
                 residual_vallin=False, **kw):
        super(MultiHeadAttention, self).__init__(**kw)
        self.querydim = querydim
        self.hdim = querydim if hdim is None else hdim
        self.valdim = querydim if valdim is None else valdim
        self.keydim = querydim if keydim is None else keydim
        self.usevallin = usevallin
        self.residual_vallin = residual_vallin
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
        weights = torch.einsum("bhd,bhsd->bhs", queries, context) / np.sqrt(context.size(-1))
        alphas = torch.softmax(weights, -1)      # bhs

        alphas = self.attn_dropout(alphas)
        values = val
        if self.value_lin is not None:
            values = self.value_lin(values)
            if self.residual_vallin:
                values = values + val
        values = values.view(values.size(0), values.size(1), self.numheads, -1).transpose(1, 2)
        red = torch.einsum("bhs,bhsd->bhd", alphas, values)
        red = red.view(red.size(0), -1)
        return red


class GRUCell(torch.nn.Module):
    def __init__(self, dim, bias=True, dropout=0., dropout_rec=0., use_layernorm=True, long_range_bias=0., **kw):
        super(GRUCell, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.long_range_bias = long_range_bias
        self.gateW = torch.nn.Linear(dim * 2, dim * 2, bias=bias)
        self.gateW.bias.data[dim:] = self.gateW.bias.data[dim:] + self.long_range_bias
        self.gateU = torch.nn.Linear(dim * 2, dim, bias=bias)
        self.sm = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout_rec = torch.nn.Dropout(dropout_rec)
        self.register_buffer("dropout_mask", torch.ones(self.dim * 2))
        self.ln = None
        self.ln2 = None
        if use_layernorm:
            self.ln = torch.nn.LayerNorm(dim * 2)
            self.ln2 = torch.nn.LayerNorm(dim * 2)

    def reset_dropout(self):
        device = self.dropout_mask.device
        ones = torch.ones(self.dim * 2, device=device)
        self.dropout_mask = self.dropout_rec(ones)

    def forward(self, x, h):
        inp = torch.cat([x, h], 1)
        if self.ln is not None:
            inp = self.ln(inp)
        inp = self.dropout(inp)
        inp = inp * self.dropout_mask[None, :]
        gates = self.gateW(inp)
        gates = gates.chunk(2, 1)
        r = torch.sigmoid(gates[0])
        z = torch.sigmoid(gates[1])
        inp = torch.cat([x, h * r], 1)
        if self.ln2 is not None:
            inp = self.ln2(inp)
        inp = self.dropout(inp)
        inp = inp * self.dropout_mask[None, :]
        u = self.gateU(inp)
        u = torch.tanh(u)
        h_new = h * z + (1 - z) * u
        return h_new


GATE_BIAS = 0
class DGRUCell(torch.nn.Module):
    zoneout_frac = .7
    def __init__(self, dim, bias=True, dropout=0., dropout_rec=0., gate_bias=GATE_BIAS, use_layernorm=True, **kw):
        super(DGRUCell, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.gateW = torch.nn.Linear(dim * 2, dim * 5, bias=bias)
        self.gateU = torch.nn.Linear(dim * 2, dim, bias=bias)
        self.sm = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout_rec = torch.nn.Dropout(dropout_rec)
        self.register_buffer("dropout_mask", torch.ones(self.dim * 2))
        self.gate_bias = gate_bias

        self.ln = None
        self.ln2 = None
        if use_layernorm:
            self.ln = torch.nn.LayerNorm(dim * 2)
            self.ln2 = torch.nn.LayerNorm(dim * 2)

    def reset_dropout(self):
        device = self.dropout_mask.device
        ones = torch.ones(self.dim * 2, device=device)
        self.dropout_mask = self.dropout_rec(ones)

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


def get_edge_frontiers(dag:dgl.DGLGraph, numechos=1):
    # must return a list over steps of lists of edge ids
    # build a dictionary of edges incoming at every node and self-edges
    u, v = dag.edges()
    u, v = list(u.cpu().numpy()), list(v.cpu().numpy())
    incoming = {i: [] for i in range(dag.number_of_nodes())}
    outgoing = {i: [] for i in range(dag.number_of_nodes())}
    selfedges = {i: [] for i in range(dag.number_of_nodes())}
    allselfedgeids = []
    edges = list(zip(u, v))
    for i, (ui, vi) in enumerate(edges):
        if ui == vi:    # self-edge
            selfedges[vi].append(i)
            allselfedgeids.append(i)
        else:
            incoming[vi].append(i)
            outgoing[ui].append(i)

    # start from nodes that don't have outgoing messages (except for self-edges)
    prev_nodeset = set()
    for k, v in outgoing.items():
        if len(v) == 0:
            prev_nodeset.add(k)

    # reverse_node frontiers:
    edge_frontiers = [[]]
    _dag = dgl.DGLGraph()
    _dag.add_nodes(dag.number_of_nodes())
    _dag.add_edges(*dag.edges())
    _dag.remove_edges(allselfedgeids)
    revnode_frontiers = dgl.topological_nodes_generator(_dag, reverse=True)
    for front in revnode_frontiers[::-1]:
        for node in list(front.cpu().numpy()):
            if len(incoming[node]) > 0:
                for edgeid in incoming[node]:
                    edge_frontiers[-1].append(edgeid)
                for selfedgeid in selfedges[node]:
                    edge_frontiers[-1].append(selfedgeid)
        if len(edge_frontiers[-1]) > 0:
            edge_frontiers.append([])
    assert len(edge_frontiers[-1]) == 0
    del edge_frontiers[-1]

    # do echos
    ret = []
    for i in range(len(edge_frontiers)+numechos-1):
        ret.append([])
        for j in range(numechos):
            if i - j >= 0 and i - j < len(edge_frontiers):
                ret[-1] += edge_frontiers[i-j]
    # these nodes will be executed last
    # run message propagation against the edges
    # node_dist = {i: 0 for i in range(dag.number_of_nodes())}
    # while len(prev_nodeset) > 0:
    #     next_nodeset = set()
    #     for nodeid in prev_nodeset:
    #         for edgeid in incoming[nodeid]:
    #             nextnode = edges[edgeid][0]
    #             node_dist[nextnode] = max(node_dist[nodeid]+1, node_dist[nextnode])
    #             next_nodeset.add(nextnode)
    #     prev_nodeset = next_nodeset
    #
    # edge_frontiers = []
    # while len(prev_nodeset) > 0:
    #     next_nodeset = set()
    #     edge_frontiers.append(set())
    #     for node in prev_nodeset:
    #         for edgeid in outgoing[node]:
    #             edge_frontiers[-1].add(edgeid)
    #             next_nodeset.add(edges[edgeid][1])
    #             for selfedgeid in selfedges[edges[edgeid][1]]:
    #                 edge_frontiers[-1].add(selfedgeid)
    #     edge_frontiers[-1] = list(edge_frontiers[-1])
    #     prev_nodeset = next_nodeset

    return ret


# TODO: use centralities as features
class MASNN(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numgnnperlayer=3, numrepspergnnlayer=1, numrepsperlayer=1, numechos=1, numrels=1, numheads=4,
                 dropout=0., dropoutrec=0.,
                 rdim=None, residual=True, masresidual=False, use_sgru=False, **kw):
        super(MASNN, self).__init__(**kw)
        self.hdim = hdim
        self.use_sgru = use_sgru
        self.layers = torch.nn.ModuleList([
            MASNNLayer(hdim, numgnnlayers=numgnnperlayer, numrepsperlayer=numrepspergnnlayer, numrels=numrels, numheads=numheads,
                       dropout=dropout, dropoutrec=dropoutrec,
                      rdim=rdim, residual=residual, masresidual=masresidual, use_sgru=use_sgru, **kw)
            for _ in range(numlayers)
        ])
        self.numrepsperlayer = numrepsperlayer
        self.numechos = numechos
        self.residual = residual

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        # self.layers[0].init_node_states(g, batsize, device)
        pass

    def forward(self, g, step=None):
        # compute dags
        # print("computing dags")
        dags1, dags2 = [], []
        for ge in dgl.unbatch(g):
            dag1, dag2 = compute_dags(ge)
            assert dag1.number_of_nodes() == dag2.number_of_nodes()
            dags1.append(dag1)
            dags2.append(dag2)
        # print("done")
        dag = dgl.batch(dags1 + dags2)
        trav = get_edge_frontiers(dag, numechos=self.numechos)
        # TODO: ensure there are no leaks between examples!
        # dags1, dags2 = dgl.batch(dags1), dgl.batch(dags2)
        # trav1 = get_edge_frontiers(dags1)
        # trav2 = get_edge_frontiers(dags2)

        if step is None:
            for layer in self.layers:
                for _ in range(self.numrepsperlayer):
                    g = layer(g, dag, trav)
                    # g = layer(g, dags1, dags2, trav1, trav2)
        else:
            # assert(step is not None)
            _step = step/self.numrepsperlayer
            _step = math.floor(_step)
            _step = min(_step, len(self.layers) - 1)
            layer = self.layers[_step]
            # g = layer(g, dags1, dags2, trav1, trav2)
            g = layer(g, dag, trav)
        return g


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


class MAS_LRGATCell(torch.nn.Module):
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0., dropoutrec=0., dropout_attn=0., rdim=None,
                 usevallin=False, residual_vallin=False, norel=False, residual=True, use_sgru=False, **kw):
        super(MAS_LRGATCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        self.zdim = hdim
        self.numrels = numrels
        self.norel = norel
        self.use_sgru = use_sgru
        self.residual = residual
        self.dropout = dropout
        self.dropoutrec = dropoutrec
        self.dropout_attn = dropout_attn
        self.usevallin = usevallin
        self.numheads = numheads
        self.residual_vallin = residual_vallin
        self.register_buffer("dropout_rec_mask", torch.ones(1, self.hdim))
        self.init_model()

    def init_model(self):
        if not self.norel:
            self.relvectors_add = torch.nn.Parameter(torch.randn(self.numrels, self.rdim))
            torch.nn.init.kaiming_uniform_(self.relvectors_add, a=math.sqrt(5))
            self.msg_ln = torch.nn.LayerNorm(self.hdim, elementwise_affine=False)
            self.relvectors = torch.nn.Embedding(self.numrels, self.hdim)
            self.relvectors2 = torch.nn.Embedding(self.numrels, self.hdim)
            torch.nn.init.ones_(self.relvectors.weight)
            torch.nn.init.zeros_(self.relvectors2.weight)

        if not self.use_sgru:
            self.gru = GRUCell(self.hdim, dropout=self.dropout, long_range_bias=2.)
        else:
            self.gru = DGRUCell(self.hdim, dropout=self.dropout)

        self.attention = MultiHeadAttention(self.hdim, self.hdim,
                                            self.hdim, self.hdim, numheads=self.numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=self.dropout_attn, usevallin=self.usevallin,
                                            residual_vallin=self.residual_vallin)

        self._dropout = torch.nn.Dropout(self.dropout)

        self.ln_att = torch.nn.LayerNorm(self.hdim)
        self.reset_params()

    def reset_params(self):
        torch.nn.init.ones_(self.relvectors.weight)
        torch.nn.init.zeros_(self.relvectors2.weight)

    def reset_dropout(self):
        self.dropout_rec_mask.fill_(1.)
        self.dropout_rec_mask = torch.dropout(self.dropout_rec_mask, self.dropoutrec, train=self.training).clamp(0, 1)

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        relvecs_add = None
        relvecs = None
        relvecs2 = None
        if self.norel:
            relvecs = torch.ones(hs.size(0), self.rdim)
            relvecs2 = torch.zeros(hs.size(0), self.rdim)
            relvecs_add = torch.ones(hs.size(0), self.rdim)
        elif "emb" in edges.data:  # if "emb" in edata, then use those
            relvecs = edges.data["emb"]
            if "emb2" in edges.data:
                relvecs2 = edges.data["emb2"]
            if "emb_add" in edges.data:
                relvecs_add = edges.data["emb_add"]
        if relvecs is None:
            relvecs = self.relvectors(edges.data["id"])
        if relvecs2 is None:
            relvecs2 = self.relvectors2(edges.data["id"])
        if relvecs_add is None:
            relvecs_add = self.relvectors_add[edges.data["id"]]

        hs = hs + relvecs_add
        hs = self.msg_ln(hs)
        hs = hs * relvecs + relvecs2
        hs = torch.nn.functional.leaky_relu(hs, 0.25)

        hs = self.dropout_rec_mask * hs

        msg = hs

        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
        red = self.attention(queries, keys, values)
        return {"red": red}

    def apply_node_func(self, nodes):
        h = self.gru(nodes.data["h"], nodes.data["red"])
        if self.residual:
            h = h + nodes.data["h"]
        h = self.dropout_rec_mask * h
        return {"h": h}

    def forward(self, dag:dgl.DGLGraph, trav, step=0):
        dag.prop_edges(trav, self.message_func, self.reduce_func, self.apply_node_func)
        return dag


class MAS_ResRGATCell(MAS_LRGATCell):
    def init_model(self):
        if not self.norel:
            self.relvectors = torch.nn.Parameter(torch.randn(self.numrels, self.rdim))
            torch.nn.init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))

            self.msg_fn = FFN(self.hdim)

        if not self.use_sgru:
            self.gru = GRUCell(self.hdim, dropout=self.dropout)
        else:
            self.gru = DGRUCell(self.hdim, dropout=self.dropout)

        self.attention = MultiHeadAttention(self.hdim, self.hdim,
                                            self.hdim, self.hdim, numheads=self.numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=self.dropout_attn, usevallin=self.usevallin,
                                            residual_vallin=self.residual_vallin)

        self._dropout = torch.nn.Dropout(self.dropout)

        self.ln_att = torch.nn.LayerNorm(self.hdim)

    def message_func(self, edges):
        hs = edges.src["h"]
        if self.norel:
            relvecs = torch.zeros(hs.size(0), self.rdim)
        elif "emb" in edges.data:  # if "emb" in edata, then use those
            relvecs = edges.data["emb"]
        else:
            relvecs = self.relvectors[edges.data["id"]]

        inps = hs + relvecs

        _x = self.msg_fn(inps)

        hs = hs + _x
        msg = hs
        return {"msg": msg, "hs": hs}


class SumMixer(torch.nn.Module):
    def __init__(self, *args, **kw):
        super(SumMixer, self).__init__()

    def forward(self, a, b):
        return a + b


class GatedMixer(torch.nn.Module):
    def __init__(self, dim, dropout=0., **kw):
        super(GatedMixer, self).__init__(**kw)
        self.dim = dim
        self.dropout = torch.nn.Dropout(dropout)
        self.linA = torch.nn.Linear(dim + dim + dim, dim + dim)
        self.nonlin = torch.nn.CELU()
        self.linB = torch.nn.Linear(dim + dim, dim + dim)
        self.linB.bias.data.fill_(2.)  # bias it to let through everything

    def forward(self, a, b):
        z = self.dropout(torch.cat([a, b, torch.max(a, b)], -1))
        z = self.linA(z)
        z = self.nonlin(z)
        gates = self.linB(z)
        gateA, gateB = torch.chunk(gates, 2, -1)
        ret = a * torch.sigmoid(gateA) + b * torch.sigmoid(gateB)
        return ret


class MASNNLayer(torch.nn.Module):
    # MASCell = MAS_LRGATCell
    MASCell = MAS_ResRGATCell
    Mixer = SumMixer
    def __init__(self, hdim, numgnnlayers=3, numrepsperlayer=1, numechos=1, numrels=10, numheads=1, dropout=0., dropoutrec=0., rdim=None, residual=True, masresidual=False, use_sgru=False, **kw):
        super(MASNNLayer, self).__init__(**kw)
        self.hdim = hdim
        self.numgnnlayers = numgnnlayers
        self.numrepsperlayer = numrepsperlayer
        self.numechos = numechos
        self.numrels = numrels
        self.numheads = numheads
        self.dropout = dropout
        self.dropoutrec = dropoutrec
        self.rdim = rdim if rdim is not None else self.hdim
        self.residual = residual
        self.masresidual = masresidual
        self.use_sgru = use_sgru

        if self.numgnnlayers > 0:
            self.gnn = self.make_gnn(self.hdim, numlayers=self.numgnnlayers, numrels=self.numrels, dropout=self.dropout, residual=self.residual, numheads=self.numheads)
        else:
            self.gnn = None

        self.masnn1 = self.MASCell(self.hdim, numrels=self.numrels, numheads=self.numheads,
                                   dropout=self.dropout, dropoutrec=self.dropoutrec,
                                   rdim=self.rdim, residual=True, use_sgru=self.use_sgru)
        self.masnn2 = self.MASCell(self.hdim, numrels=self.numrels, numheads=self.numheads,
                                   dropout=self.dropout, dropoutrec=self.dropoutrec,
                                   rdim=self.rdim, residual=True, use_sgru=self.use_sgru)

        self.mixer1 = self.Mixer(self.hdim, dropout=self.dropout)
        self.mixer2 = self.Mixer(self.hdim, dropout=self.dropout)

        self.diremb = torch.nn.Embedding(2, self.hdim)
        torch.nn.init.zeros_(self.diremb.weight)

    def init_node_states(self, g, batsize, device):
        self.gnn.init_node_states(g, batsize, device)

    def reset_dropout(self):
        self.masnn1.reset_dropout()
        self.masnn2.reset_dropout()

    def make_gnn(self, hdim, numlayers=3, numrels=10, dropout=0., residual=True, numheads=1):
        gnns = [LRTMCell(hdim, numrels=numrels, dropout=dropout, numheads=numheads) for _ in range(numlayers)]
        gnn = torch.nn.ModuleList(gnns)
        return gnn

    # def forward(self, g, dag1=None, dag2=None, trav1=None, trav2=None):
    def forward(self, g:dgl.DGLGraph, dag:dgl.DGLGraph=None, trav=None):
        if dag is None:
            # compute dags
            dags1, dags2 = [], []
            for ge in dgl.unbatch(g):
                dag1, dag2 = compute_dags(ge)
                assert dag1.number_of_nodes() == dag2.number_of_nodes()
                dags1.append(dag1)
                dags2.append(dag2)
            dag = dgl.batch(dags1 + dags2)
            trav = get_edge_frontiers(dag, numechos=self.numechos)

        # run normal gnn first
        if self.gnn is not None:
            for gnnlayer in self.gnn:
                for _ in range(self.numrepsperlayer):
                    g = gnnlayer(g)
            h = g.ndata["h"]

        if self.numechos > 0:

            _h = g.ndata["h"]
            assert g.number_of_nodes()*2 == dag.number_of_nodes()
            fwdvec, revvec = self.diremb(torch.tensor([0], device=_h.device, dtype=torch.long)), \
                             self.diremb(torch.tensor([1], device=_h.device, dtype=torch.long))

            dag.ndata["h"] = torch.cat([_h + fwdvec, _h + revvec], 0)
            dag = self.masnn1(dag, trav)
            h = dag.ndata["h"]
            # h = h[:len(h)//2] + h[len(h)//2:]   # back to number of normal nodes
            h = self.mixer1(h[:len(h) // 2], h[len(h) // 2:])
            if self.masresidual:
                h = h + _h

            _h = h
            dag.ndata["h"] = torch.cat([h + fwdvec, h + revvec], 0)
            dag = self.masnn2(dag, trav)
            h = dag.ndata["h"]
            # h = h[:len(h)//2] + h[len(h)//2:]   # back to number of normal nodes
            h = self.mixer2(h[:len(h) // 2], h[len(h) // 2:])
            if self.masresidual:
                h = h + _h

        g.ndata["h"] = h
        return g


class Readout(torch.nn.Module):
    def __init__(self, mode="mean", **kw):
        super(Readout, self).__init__(**kw)
        self.mode = mode

    def forward(self, g, attr):
        if "sum" in self.mode.split("-"):
            ret = dgl.sum_nodes(g, attr)
        elif "max" in self.mode.split("-"):
            ret = dgl.max_nodes(g, attr)
        elif "mean" in self.mode.split("-"):
            ret = dgl.mean_nodes(g, attr)
        return ret


class MaskedReadout(Readout):
    def __init__(self, mode="mean", maskattr="mask", **kw):
        super(MaskedReadout, self).__init__(mode=mode, **kw)
        self.maskattr = maskattr

    def forward(self, g:dgl.DGLGraph, attr, maskattr=None):
        maskattr = maskattr if maskattr is not None else self.maskattr
        mask = g.ndata[maskattr] == 1
        _g = g.local_var()
        if "sum" in self.mode.split("-"):
            _g.ndata[attr] = g.ndata[attr] * mask[:, None].float()
            ret = dgl.sum_nodes(_g, attr)
        elif "max" in self.mode.split("-"):
            _g.ndata[attr] = g.ndata[attr] - (1 - mask[:, None].float()) * 99999
            ret = dgl.max_nodes(_g, attr)
        elif "mean" in self.mode.split("-"):
            _g.ndata[attr] = g.ndata[attr] * mask[:, None].float()
            _g.ndata["poolmask"] = mask[:, None].float()
            s = dgl.sum_nodes(_g, attr)
            mask_sum = dgl.sum_nodes(_g, "poolmask")
            ret = s/mask_sum
        return ret


def tst_send_and_receive():
    g = dgl.DGLGraph()
    g.add_nodes(9, data={"id": torch.arange(9)})
    # g.add_edges([0, 1, 2, 3, 4, 6, 5, 5, 3, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    #             [1, 5, 5, 2, 3, 7, 7, 8, 8, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8], data={"id": torch.arange(19)})
    g.add_edges([1, 5, 5, 2, 3, 7, 7, 8, 8, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                [0, 1, 2, 3, 4, 6, 5, 5, 3, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8], data={"id": torch.arange(19)})

    edge_frontiers = get_edge_frontiers(g)
    print(edge_frontiers)
    def msg_f(edges):
        ret = {"m": edges.data["id"]}
        return ret
    def red_f(nodes):
        ret = {"h": nodes.mailbox["m"].max(-1)[0]}
        return ret
    def apl_f(nodes):
        ret = {"h": nodes.data["h"]}
        return ret
    g.prop_edges(edge_frontiers, msg_f, red_f, apl_f)


if __name__ == '__main__':
    tst_send_and_receive()
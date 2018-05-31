"""
Minjie's code
"""

from __future__ import division

from collections import namedtuple
import math
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

GlobalIndex = namedtuple('GlobalIndex',
    ['lcl_v_glbid', 'edge_mask_src', 'edge_mask_tgt',
     'srcneigh_v_glbid', 'srcneigh_row', 'srcneigh_col', 'src_adj',
     'tgtneigh_v_glbid', 'tgtneigh_row', 'tgtneigh_col', 'tgt_adj'])
CrossIndex = namedtuple('CrossIndex', ['src_v_lclid', 'tgt_v_lclid'])

class Partitioner(object):
    def __init__(self, graph_or_adj, K, rng=None):
        """Params:
        graph_or_adj - either a nx.Graph instance of an adjcency matrix
                       in scipy.sparse.coo_matrix format
        K   - the number of partitions.
        rng - the given random number generator
        """
        self._graph = None
        self._adj = None
        if isinstance(graph_or_adj, nx.Graph):
            self._graph = graph_or_adj
        else:
            self._adj = graph_or_adj
        self.K = K
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng
        self.vassign = None
        self.eassign = None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = nx.from_scipy_sparse_matrix(self._adj, create_using=nx.DiGraph())
        return self._graph

    @property
    def adj(self):
        if self._adj is None:
            self._adj = nx.adjacency_matrix(self._graph).tocoo()
        return self._adj

    @property
    def num_nodes(self):
        if self._graph is not None:
            return self._graph.number_of_nodes()
        else:
            return self.adj.shape[0]

    @property
    def num_edges(self):
        if self._graph is not None:
            return self._graph.number_of_edges()
        else:
            return len(self.adj.data)

    def partition(self):
        raise RuntimeError('Not implemented.')

    def pprint(self):
        #raise NotImplementedError()
        ecut_flag = len(self.vassign.shape) == 1
        vloads = [0 for _ in range(self.K)]
        eloads = [0 for _ in range(self.K)]
        if ecut_flag:
            for v in range(self.num_nodes):
                vloads[self.vassign[v]] += 1
        else:
            for v in range(self.num_nodes):
                for k in range(self.K):
                    if self.vassign[v, k] == 1:
                        vloads[k] += 1
        for e in range(self.num_edges):
            eloads[self.eassign[e]] += 1

        def entropy(x_list):
            z = sum(x_list)
            p_list = [x / z for x in x_list]
            return -sum(p * math.log(p) for p in p_list)

        print('#Verts:', vloads, entropy(vloads))
        print('#Edges:', eloads, entropy(eloads))
        num_cross_edges = 0
        for i, (u, v) in enumerate(zip(self.adj.row, self.adj.col)):
            if ecut_flag:
                if self.vassign[u] != self.vassign[v]:
                    num_cross_edges += 1
            else:
                k = self.eassign[i]
                if self.vassign[u, k] != 1 or self.vassign[v, k] != 1:
#                   print('Edge', i, 'u', u, 'v', v)
#                   print(k, self.vassign[u,:], self.vassign[v,:])
                    num_cross_edges += 1
#       print('#Cross-edges:', num_cross_edges)
        if ecut_flag:
            num_vert_rep = [0 for i in range(self.num_nodes)]
        else:
            num_vert_rep = np.sum(self.vassign, axis=1)
        print('#Replicate-factor-avg:', np.average(num_vert_rep))
        print('#Replicate-factor-max:', np.max(num_vert_rep))
        num_rep_dist = [0 for i in range(self.K + 1)]
        for v, r in enumerate(num_vert_rep):
            num_rep_dist[r] += 1
        print('#Replicate-factor-dist:', num_rep_dist)

    def build_index(self):
        # TODO: edgeset index
        edge_assign_src = self.vassign[self.adj.col]  # assignment of each edge src endpoint
        edge_assign_tgt = self.vassign[self.adj.row]  # assignment of each edge tgt endpoint
        v_glbid_to_lclid = np.zeros((self.num_nodes,), dtype=np.int32) - 1
        tmp_space = np.zeros((self.num_nodes,), dtype=np.int32)
        def _build_glb_index(k):
            lcl_v_glbid = np.where(self.vassign == k)[0]
            lcl_v_lclid = np.arange(lcl_v_glbid.shape[0])
            lcl_size = lcl_v_glbid.shape[0]
            v_glbid_to_lclid[lcl_v_glbid] = lcl_v_lclid
            if k == 0:
                print('Local vset size:', lcl_v_glbid.shape)
            #lcl_e_glbid = np.where(self.eassign == k)[0]
            edge_mask_src = (edge_assign_src == k)
            edge_mask_tgt = (edge_assign_tgt == k)
            edge_idx_src = np.where(edge_mask_src)[0]  # all out-going edges from partition k
            edge_idx_tgt = np.where(edge_mask_tgt)[0]  # all in-coming edges to partition k
            # src neighbors
            srcneigh_row = self.adj.row[edge_idx_tgt]
            srcneigh_col = self.adj.col[edge_idx_tgt]
            srcneigh_data = self.adj.data[edge_idx_tgt]
            srcneigh_v_glbid = np.unique(srcneigh_col)
            srcneigh_size = srcneigh_v_glbid.shape[0]
            tmp_space[srcneigh_v_glbid] = np.arange(srcneigh_size)
            # create adj from srcneigh->lcl
            src_adj = sp.coo_matrix(
                (srcneigh_data, (v_glbid_to_lclid[srcneigh_row], tmp_space[srcneigh_col])),
                shape=(lcl_size, srcneigh_size))
            if k == 0:
                print('Src neigh vset size:', srcneigh_v_glbid.shape)
                print('Src neigh eset size:', srcneigh_row.shape)
            # tgt neighbors
            tgtneigh_row = self.adj.row[edge_idx_src]
            tgtneigh_col = self.adj.col[edge_idx_src]
            tgtneigh_data = self.adj.data[edge_idx_src]
            tgtneigh_v_glbid = np.unique(tgtneigh_row)
            tgtneigh_size = tgtneigh_v_glbid.shape[0]
            tmp_space[tgtneigh_v_glbid] = np.arange(tgtneigh_size)
            # create adj from tgtneigh->lcl
            tgt_adj = sp.coo_matrix(
                (tgtneigh_data, (tmp_space[tgtneigh_row], v_glbid_to_lclid[tgtneigh_col])),
                shape=(tgtneigh_size, lcl_size)).T
            if k == 0:
                print('Tgt neigh vset size:', tgtneigh_v_glbid.shape)
                print('Tgt neigh eset size:', tgtneigh_row.shape)
            return GlobalIndex(lcl_v_glbid, edge_mask_src, edge_mask_tgt,
                               srcneigh_v_glbid, srcneigh_row, srcneigh_col, src_adj,
                               tgtneigh_v_glbid, tgtneigh_row, tgtneigh_col, tgt_adj)
        glb_index = []
        for k in range(self.K):
            glb_index.append(_build_glb_index(k))
        # sanity check
        assert np.all(v_glbid_to_lclid >= 0)
        def _build_cross_index(k1, k2):
            # find all edges from k1 to k2
            edge_mask_k1k2 = np.logical_and(glb_index[k1].edge_mask_src,
                                            glb_index[k2].edge_mask_tgt)
            edge_idx_k1k2 = np.where(edge_mask_k1k2)[0]
            # all the tgt endpoint vertices in k2
            k2tgt_v_glbid = np.unique(self.adj.row[edge_idx_k1k2])
            tgt_v_lclid = v_glbid_to_lclid[k2tgt_v_glbid]
            # all the src endpoint vertices in k1
            k1src_v_glbid = np.unique(self.adj.col[edge_idx_k1k2])
            src_v_lclid = v_glbid_to_lclid[k1src_v_glbid]
            if k1 == 0:
                print('k1 vset size from %d->%d' % (k1, k2), src_v_lclid.shape)
                print('k2 vset size from %d->%d' % (k1, k2), tgt_v_lclid.shape)
            return CrossIndex(src_v_lclid, tgt_v_lclid)
        cross_index = []
        for k1 in range(self.K):
            cross_index.append([])
            for k2 in range(self.K):
                cross_index[k1].append(_build_cross_index(k1, k2))
        # sanity check
        for k1 in range(self.K):
            sn_vsize = glb_index[k1].srcneigh_v_glbid.shape[0]
            assert sn_vsize == sum([cross_index[k2][k1].src_v_lclid.shape[0] for k2 in range(self.K)])
            tn_vsize = glb_index[k1].tgtneigh_v_glbid.shape[0]
            assert tn_vsize == sum([cross_index[k1][k2].tgt_v_lclid.shape[0] for k2 in range(self.K)])
        return glb_index, cross_index

class RandomEdgeCut(Partitioner):
    def partition(self):
        self.vassign = self.rng.randint(low=0, high=self.K, size=(self.num_nodes,))
        # toss a coin to decide whether the edge should be assigned to source node or target node.
        src_mask = self.rng.randint(low=0, high=2, size=(self.num_edges,))
        tgt_mask = 1 - src_mask
        self.eassign = src_mask * self.vassign[self.adj.row] + tgt_mask * self.vassign[self.adj.col]

class IngressEdgeCut(Partitioner):
    """Assign edges to their source vertices."""
    def partition(self):
        self.vassign = self.rng.randint(low=0, high=self.K, size=(self.num_nodes,))
        self.eassign = self.vassign[self.adj.row]

class EgressEdgeCut(Partitioner):
    """Assign edges to their destination vertices."""
    def partition(self):
        self.vassign = self.rng.randint(low=0, high=self.K, size=(self.num_nodes,))
        self.eassign = self.vassign[self.adj.col]

class RandomVertexCut(Partitioner):
    def partition(self):
        self.vassign = np.zeros((self.num_nodes, self.K), dtype=np.int32)
        self.eassign = self.rng.randint(low=0, high=self.K, size=(self.num_edges,))
        self.vassign[self.adj.row, self.eassign] = 1
        self.vassign[self.adj.col, self.eassign] = 1

def _assign_min_loads(mv, loads):
    if mv is None:
        mv = range(len(loads))
    min_loads = max(loads) + 1
    arg_min_loads = -1
    for k in mv:
        if loads[k] < min_loads:
            min_loads = loads[k]
            arg_min_loads = k
    return arg_min_loads

class GreedyVertexCut(Partitioner):
    def partition(self):
        self.vassign = np.zeros((self.graph.number_of_nodes(), self.K), dtype=np.int32)
        self.eassign = np.zeros((self.graph.number_of_edges(),), dtype=np.int32)
        loads = [0 for _ in range(self.K)]
        unassigned = [d for _, d in self.graph.degree().items()]
        edge_order = list(range(self.graph.number_of_edges()))
        self.rng.shuffle(edge_order)
        edges = list(self.graph.edges())
        for i in edge_order:
            u, v = edges[i]
            mu = set(np.where(self.vassign[u,:] == 1)[0].tolist())
            mv = set(np.where(self.vassign[v,:] == 1)[0].tolist())
            muv = mu.intersection(mv)
            if len(muv) != 0:
                # case 1: Assign to the intersection.
                k = _assign_min_loads(muv, loads)
            elif len(mu) != 0 and len(mv) != 0:
                # case 2: Both u and v have been assigned. Assign to the vertex with 
                #         the most unassigned edges.
                if unassigned[u] < unassigned[v]:
                    k = _assign_min_loads(mv, loads)
                else:
                    k = _assign_min_loads(mu, loads)
            elif len(mu) != 0:
                # case 3: Only one of the vertices has been assigned.
                k = _assign_min_loads(mu, loads)
            elif len(mv) != 0:
                # case 3: Only one of the vertices has been assigned.
                k = _assign_min_loads(mv, loads)
            else:
                # case 4: Neither vertex has been assigned. Assign to least loaded machine.
                k = _assign_min_loads(None, loads)
            self.eassign[i] = k
            self.vassign[u, k] = 1
            self.vassign[v, k] = 1
            loads[k] += 1
            unassigned[u] -= 1
            unassigned[v] -= 1

class OracleRandomEdgeCut(Partitioner):
    def __init__(self, graph_or_adj, K, gt, rng=None):
        super(OracleRandomEdgeCut, self).__init__(graph_or_adj, K, rng)
        self.ground_truth = gt

    def partition(self):
        self.vassign = np.array(self.ground_truth, dtype=np.int32)
        # toss a coin to decide whether the edge should be assigned to source node or target node.
        src_mask = self.rng.randint(low=0, high=2, size=(self.num_edges,))
        tgt_mask = 1 - src_mask
        self.eassign = src_mask * self.vassign[self.adj.row] + tgt_mask * self.vassign[self.adj.col]

class OracleIngressEdgeCut(Partitioner):
    def __init__(self, graph_or_adj, K, gt, rng=None):
        super(OracleIngressEdgeCut, self).__init__(graph_or_adj, K, rng)
        self.ground_truth = gt

    def partition(self):
        self.vassign = np.array(self.ground_truth, dtype=np.int32)
        self.eassign = self.vassign[self.adj.row]

class OracleEgressEdgeCut(Partitioner):
    def __init__(self, graph_or_adj, K, gt, rng=None):
        super(OracleEgressEdgeCut, self).__init__(graph_or_adj, K, rng)
        self.ground_truth = gt

    def partition(self):
        self.vassign = np.array(self.ground_truth, dtype=np.int32)
        self.eassign = self.vassign[self.adj.col]

class OracleRandomVertexCut(Partitioner):
    def __init__(self, graph, K, gt, rng=None):
        super(OracleRandomVertexCut, self).__init__(graph, K, rng)
        self.ground_truth = gt

    def partition(self):
        # first assign edges randomly to either source or destination vertices
        self.vassign = np.array(self.ground_truth, dtype=np.int32)
        # toss a coin to decide whether the edge should be assigned to source node or target node.
        src_mask = self.rng.randint(low=0, high=2, size=(self.num_edges,))
        tgt_mask = 1 - src_mask
        self.eassign = src_mask * self.vassign[self.adj.row] + tgt_mask * self.vassign[self.adj.col]
        # replicate vertices to associate edges
        va = np.zeros((self.num_nodes, self.K), dtype=np.int32)
        va[self.adj.row, self.eassign] = 1
        va[self.adj.col, self.eassign] = 1
        self.vassign = va

class OracleIngressVertexCut(Partitioner):
    def __init__(self, graph, K, gt, rng=None):
        super(OracleIngressVertexCut, self).__init__(graph, K, rng)
        self.ground_truth = gt

    def partition(self):
        self.vassign = np.array(self.ground_truth, dtype=np.int32)
        self.eassign = self.vassign[self.adj.row]
        # replicate vertices to associate edges
        va = np.zeros((self.num_nodes, self.K), dtype=np.int32)
        va[self.adj.row, self.eassign] = 1
        va[self.adj.col, self.eassign] = 1
        self.vassign = va

class OracleEgressVertexCut(Partitioner):
    def __init__(self, graph, K, gt, rng=None):
        super(OracleEgressVertexCut, self).__init__(graph, K, rng)
        self.ground_truth = gt

    def partition(self):
        self.vassign = np.array(self.ground_truth, dtype=np.int32)
        self.eassign = self.vassign[self.adj.col]
        # replicate vertices to associate edges
        va = np.zeros((self.num_nodes, self.K), dtype=np.int32)
        va[self.adj.row, self.eassign] = 1
        va[self.adj.col, self.eassign] = 1
        self.vassign = va

class OracleGreedyVertexCut(Partitioner):
    def __init__(self, graph, K, gt, rng=None):
        super(OracleGreedyVertexCut, self).__init__(graph, K, rng)
        self.ground_truth = gt

    def partition(self):
        # first assign edges to the vertices with lower degrees
        self.vassign = np.array(self.ground_truth, dtype=np.int32)
        self.eassign = np.zeros((self.num_edges,), dtype=np.int32)
        for i, (u, v) in enumerate(self.graph.edges()):
            if self.graph.degree(u) < self.graph.degree(v):
                self.eassign[i] = self.vassign[u]
            else:
                self.eassign[i] = self.vassign[v]
        # replicate vertices to associate edges
        va = np.zeros((self.graph.number_of_nodes(), self.K), dtype=np.int32)
        for v, k in enumerate(self.vassign):
            va[v, k] = 1
        for i, (u, v) in enumerate(self.graph.edges()):
            k = self.eassign[i]
            va[u, k] = 1
            va[v, k] = 1
        self.vassign = va

def partition(partitioner, graph_or_adj, K, gt=None, rng=None, pprint=False):
    if partitioner == 'random-edge-cut':
        part = RandomEdgeCut(graph_or_adj, K, rng)
    elif partitioner == 'ingress-edge-cut':
        part = IngressEdgeCut(graph_or_adj, K, rng)
    elif partitioner == 'egress-edge-cut':
        part = EgressEdgeCut(graph_or_adj, K, rng)
    elif partitioner == 'random-vertex-cut':
        part = RandomVertexCut(graph_or_adj, K, rng)
    elif partitioner == 'greedy-vertex-cut':
        part = GreedyVertexCut(graph_or_adj, K, rng)
    elif partitioner == 'oracle-random-edge':
        part = OracleRandomEdgeCut(graph_or_adj, K, gt, rng)
    elif partitioner == 'oracle-ingress-edge':
        part = OracleIngressEdgeCut(graph_or_adj, K, gt, rng)
    elif partitioner == 'oracle-egress-edge':
        part = OracleEgressEdgeCut(graph_or_adj, K, gt, rng)
    elif partitioner == 'oracle-random-vertex':
        part = OracleRandomVertexCut(graph_or_adj, K, gt, rng)
    elif partitioner == 'oracle-ingress-vertex':
        part = OracleIngressVertexCut(graph_or_adj, K, gt, rng)
    elif partitioner == 'oracle-egress-vertex':
        part = OracleEgressVertexCut(graph_or_adj, K, gt, rng)
    elif partitioner == 'oracle-greedy-vertex':
        part = OracleGreedyVertexCut(graph_or_adj, K, gt, rng)
    else:
        raise ValueError('Unknown partitioner "%s".' % partitioner)
    print('>>Create partitioner:', type(part))
    part.partition()
    if pprint:
       part.pprint()
    return part

ALL_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
def plot_partition(partitioner, graph, name='cluster-result'):
    edges = list(graph.edges())
    x = [u for u, _ in edges]
    y = [v for _, v in edges]
    c = [ALL_COLORS[partitioner.eassign[i]] for i, _ in enumerate(edges)]
    plt.scatter(x, y, s=1, marker='.', c=c)
    plt.savefig('%s.pdf' % name)
    plt.clf()

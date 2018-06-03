import networkx as nx
import numpy as np
import tensorflow as tf
# import torch as th


def read_edgelist(f, connected=False, create_using=None, fill=False):
    if create_using is None:
        create_using = nx.Graph()
    g = nx.read_edgelist(f, create_using=create_using, nodetype=int)

    if connected:
        if isinstance(g, nx.DiGraph):
            g = nx.subgraph(g, next(nx.weakly_connected_components(g))).copy()
        elif isinstance(g, nx.Graph):
            g = nx.subgraph(g, next(nx.connected_components(g))).copy()
        else:
            assert False

    if fill:
        nodes = list(g.nodes())
        if g.number_of_nodes() < max(g.nodes()) - min(g.nodes()) + 1:
            nodes.sort()
            idx = np.where(np.array(nodes[:-1]) + 1 != np.array(nodes[1:]))[0]
            for i in idx:
                i = int(i)
                for n in range(nodes[i], nodes[i + 1]):
                    g.add_node(n)

    return g


def sparse_sp2tf(matrix):
    coo = matrix.tocoo()
    idx = [[i, j] for i, j in zip(coo.row, coo.col)]
    return tf.SparseTensor(idx, coo.data.tolist(), coo.shape)


def sparse_sp2th(matrix):
    coo = matrix.tocoo()
    rows = th.from_numpy(coo.row).long().view(1, -1)
    cols = th.from_numpy(coo.col).long().view(1, -1)
    data = th.from_numpy(coo.data).float()
    return th.sparse.FloatTensor(th.cat((rows, cols), 0), data, coo.shape)


def onehot(x, d):
    """
    Parameters
    ----------
    x : (n,) or (n, 1)
    """

    x = x.unsqueeze(1) if x.dim() == 1 else x
    ret = th.zeros(x.size(0), d)
    is_cuda = x.is_cuda
    x = x.cpu()
    ret.scatter_(1, x, 1)
    return ret.cuda() if is_cuda else ret

import networkx as nx
import numpy as np
import torch as th


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


def read_edgelist(f):
    g = nx.read_edgelist(f, nodetype=int)
    nodes = list(g.nodes())
    if g.number_of_nodes() < max(g.nodes()) - min(g.nodes()) + 1:
        nodes.sort()
        idx = np.where(np.array(nodes[:-1]) + 1 != np.array(nodes[1:]))[0]
        for i in idx:
            i = int(i)
            for n in range(nodes[i], nodes[i + 1]):
                g.add_node(n)
    return g

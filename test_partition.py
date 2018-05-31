"""
Minjie's code
"""

import snap_utils

#snap_utils.convert_wiki_graph()
#snap_utils.load_wiki_graph()
#snap_utils.convert_wiki_categories()
#snap_utils.make_fake_wiki_features()

#snap_utils.convert_youtube_graph()
#snap_utils.convert_youtube_cmty()

import numpy as np
import networkx as nx
import sbm
import partition as pttn

K = 10
N = 1000
rng = np.random.RandomState(42)

# a = 10
# b = 2

'''
ssbm = sbm.SSBM(N, K, a=7, b=2, rng=rng)
ssbm.generate()
#ssbm.plot()
g = ssbm.graph
'''

import sys
import my

path = 'data/' + sys.argv[1]
g = my.read_edgelist(path, nx.DiGraph())

print('#Total verts:', g.number_of_nodes())
print('#Total edges:', g.number_of_edges())
adj = nx.adjacency_matrix(g)
#print('Connected?', nx.is_connected(g))
#gt = ssbm.node2comm
#p = pttn.partition('random-edge-cut', g, K, rng, pprint=True)
#p = pttn.partition('ingress-edge-cut', g, K, rng, pprint=True)
#p = pttn.partition('egress-edge-cut', g, K, rng, pprint=True)
#p = pttn.partition('random-vertex-cut', g, K, rng, pprint=True)
p = pttn.partition('greedy-vertex-cut', g, K, rng, pprint=True)
#p = pttn.partition('oracle-random-edge', g, K, gt, rng, pprint=True)
#p = pttn.partition('oracle-ingress-edge', g, K, gt, rng, pprint=True)
#p = pttn.partition('oracle-egress-edge', g, K, gt, rng, pprint=True)
#p = pttn.partition('oracle-random-vertex', g, K, gt, rng, pprint=True)
##p = pttn.partition('oracle-ingress-vertex', g, K, gt, rng, pprint=True)
##p = pttn.partition('oracle-egress-vertex', g, K, gt, rng, pprint=True)
##p = pttn.partition('oracle-greedy-vertex', g, K, gt, rng, pprint=True)
#pttn.plot_partition(p, g)

'''
glb_index, cross_index = p.build_index()

D = 10
x = rng.randn(N, 10).astype(np.float32)
y = adj.dot(x)

x_dev = [x[glb_index[k].lcl_v_glbid, :] for k in range(K)]
y_dev = []
for k1 in range(K):
    x_src = []
    for k2 in range(K):
         x_src.append(x_dev[k2][cross_index[k2][k1].src_v_lclid, :])
    s = np.vstack(x_src)
    y_dev.append(glb_index[k1].src_adj.dot(s))
yy = np.vstack(y_dev)

#x00 = x_dev[0][cross_index[0][0].src_v_lclid, :]
#x01 = x_dev[1][cross_index[1][0].src_v_lclid, :]
#x0_n = np.vstack([x00, x01])
#y0 = glb_index[0].src_adj.dot(x0_n)
#
#x10 = x_dev[0][cross_index[0][1].src_v_lclid]
#x11 = x_dev[1][cross_index[1][1].src_v_lclid]
#x1_n = np.vstack([x10, x11])
#y1 = glb_index[1].src_adj.dot(x1_n)
#
#yy = np.vstack([y0, y1])

print(np.sum(y - yy))
'''

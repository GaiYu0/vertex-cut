import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys

def convert_wiki_graph():
    dict_of_lists = {}
    with open('data/snap/wiki-topcats/wiki-topcats.txt', 'r') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print('%d lines processed' % i)
            src, tgt = line.strip().split(' ')
            src = int(src)
            tgt = int(tgt)
            if not src in dict_of_lists:
                dict_of_lists[src] = []
            dict_of_lists[src].append(tgt)
    print('Finished loading.')
    graph = nx.DiGraph(dict_of_lists)
    nx.write_gpickle(graph, 'data/snap/wiki-topcats/graph.pkl')
    print('Finished pickling.')

def convert_wiki_categories():
    num_nodes = 1791489
    node2cat = [[] for i in range(num_nodes)]
    num_cats = 0
    with open('data/snap/wiki-topcats/wiki-topcats-categories.txt', 'r') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print('%d lines processed' % i)
            splits = line.strip().split(' ')
            for s in splits[1:]:
                node2cat[int(s)].append(i)
            num_cats += 1
    print('#Categories:', num_cats)
    # Do a graph clustering down-sample number of categories to 10.
    # The distance is defined as the number of overlapped pages.
    first_cat = [x[0] for x in node2cat]
    #print(first_cat)
    print(len(set(first_cat)))
    with open('data/snap/wiki-topcats/cats.pkl', 'wb') as f:
        pkl.dump(first_cat, f)

def make_fake_wiki_features(num_features=512, shard_size=10000):
    num_nodes = 1791489
    num_shards = num_nodes // shard_size + 1
    for i in range(num_shards):
        shard_st = i * shard_size
        shard_ed = min((i + 1) * shard_size, num_nodes)
        feats = np.random.randn((shard_ed - shard_st), num_features).astype(np.float32)
        np.save('data/snap/wiki-topcats/feats/feat-%05d' % i, feats)
        print('finished shard %d' % i)

def load_wiki_graph():
    graph = nx.read_gpickle('data/snap/wiki-topcats/graph.pkl')
    print('Finished unpickling.')
    print('#Nodes:', graph.number_of_nodes())
    print('#Edges:', graph.number_of_edges())
    return graph

def convert_youtube_graph():
    dict_of_lists = {}
    with open('data/snap/com-youtube/com-youtube.ungraph.txt', 'r') as f:
        for i, line in enumerate(f):
            if i < 4:
                # Skip the first four lines.
                continue
            if i % 1000 == 0:
                print('%d lines processed' % i)
            src, tgt = line.strip().split('\t')
            src = int(src)
            tgt = int(tgt)
            if not src in dict_of_lists:
                dict_of_lists[src] = []
            dict_of_lists[src].append(tgt)
    print('Finished loading.')
    graph = nx.DiGraph(dict_of_lists)
    nx.write_gpickle(graph, 'data/snap/com-youtube/graph.pkl')
    print('Finished pickling.')

def convert_youtube_cmty():
    num_nodes = 1134890
    node2cmt = np.zeros([num_nodes], dtype=np.int32) - 1
    num_cmt = 0
    with open('data/snap/com-youtube/com-youtube.all.cmty.txt', 'r') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print('%d lines processed' % i)
            splits = line.strip().split('\t')
            for s in splits:
                assert node2cmt[int(s)] < 0
                node2cmt[int(s)] = num_cmt
            num_cmt += 1
    print(node2cmt)
    print(np.where(node2cmt < 0)[0].shape)

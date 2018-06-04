
# coding: utf-8

# In[1]:


import argparse
import networkx as nx
import numpy as np
import scipy as sp
import tensorflow as tf
import my
import objectives
import partition as pttn


# In[2]:


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.enable_eager_execution(config)


# In[3]:


'''
args = argparse.Namespace()
args.device = '/cpu:0'
args.graph = 'soc-Epinions1'
args.lambda_node = 1.0
args.lambda_edge = 1.0
args.lambda_entropy = 1.0
args.n_iterations = 500
args.n_machines = 10
args.partitioner = 'greedy'
# args.partitioner = 'random'
'''

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='/cpu:0')
parser.add_argument('--graph', type=str, default='soc-Epinions1')
parser.add_argument('--lambda_node', type=float, default=1)
parser.add_argument('--lambda_edge', type=float, default=1)
parser.add_argument('--lambda_entropy', type=float, default=1)
parser.add_argument('--n-iterations', type=int, default=500)
parser.add_argument('--n-machines', type=int, default=10)
parser.add_argument('--partitioner', type=str, default='greedy')
args = parser.parse_args()

keys = sorted(vars(args).keys())
run_id = 'oracle-' + '-'.join('%s-%s' % (key, str(getattr(args, key))) for key in keys if key != 'device')
writer = tf.contrib.summary.create_file_writer('runs/' + run_id)
writer.set_as_default()


# In[4]:


g = my.read_edgelist('data/' + args.graph, fill=True)
adj = nx.adj_matrix(g)
adj = sp.sparse.triu(adj)


# In[5]:


rng = np.random.RandomState(42)
partitioner = {
    'greedy' : pttn.GreedyVertexCut,
    'random' : pttn.RandomVertexCut,
}[args.partitioner](adj, args.n_machines, rng)


# In[6]:


partitioner.partition()
z = tf.one_hot(tf.constant(partitioner.eassign), args.n_machines)


# In[7]:


with tf.device(args.device):
    objective = objectives.Objective(adj, args.n_machines, args.lambda_node, args.lambda_edge, args.lambda_entropy)


# In[8]:


with tf.device(args.device):
    r, b_node, b_edge, entropy = objective(z, probability=True)
    rslt = r + args.lambda_node * b_node + args.lambda_edge * b_edge + args.lambda_entropy * entropy
    print('%f %f %f %f' % (r, b_node, b_edge, entropy))


# In[9]:


global_step = tf.train.get_or_create_global_step()
for i in range(args.n_iterations):
    global_step.assign_add(1)
    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('objective', rslt)
        tf.contrib.summary.scalar('replication-factor', r)
        tf.contrib.summary.scalar('node-balancedness', b_node)
        tf.contrib.summary.scalar('edge-balancedness', b_edge)
        tf.contrib.summary.scalar('entropy', entropy)


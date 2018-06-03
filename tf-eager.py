
# coding: utf-8

# In[ ]:


import argparse
import networkx as nx
import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow.contrib.eager as eager
import my
import objectives
import tf_gnn


# In[ ]:


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.enable_eager_execution(config)


# In[ ]:


'''
args = argparse.Namespace()
args.depth = 10
args.device = '/cpu:0'
# args.device = '/device:GPU:0'
args.graph = 'soc-Epinions1'
# args.graph = 'soc-Slashdot0811'
# args.graph = 'soc-Slashdot0902'
args.lambda_node = 1.0
args.lambda_edge = 1.0
args.lambda_entropy = 1.0
args.n_features = 16
args.n_iterations = 500
args.n_machines = 10
args.radius = 3
'''

parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, default=30)
parser.add_argument('--device', type=str, default='/cpu:0')
parser.add_argument('--graph', type=str, default='soc-Epinions1')
parser.add_argument('--lambda_node', type=float, default=1)
parser.add_argument('--lambda_edge', type=float, default=1)
parser.add_argument('--lambda_entropy', type=float, default=1)
parser.add_argument('--n-features', type=int, default=16)
parser.add_argument('--n-iterations', type=int, default=500)
parser.add_argument('--n-machines', type=int, default=10)
parser.add_argument('--radius', type=int, default=3)
args = parser.parse_args()

keys = sorted(vars(args).keys())
run_id = '-'.join('%s-%s' % (key, str(getattr(args, key))) for key in keys if key != 'device')
writer = tf.contrib.summary.create_file_writer('runs/' + run_id)
writer.set_as_default()


# In[ ]:


g = my.read_edgelist('data/' + args.graph, fill=True)
adj = nx.adj_matrix(g)
# x = sp.sparse.random(1000, 1000, 1e-5, data_rvs=lambda shape: np.ones(shape))
# adj = (x + x.transpose()).minimum(1)


# In[ ]:


with tf.device(args.device):
    objective = objectives.Objective(adj, args.n_machines, args.lambda_node, args.lambda_edge, args.lambda_entropy)
    gnn = tf_gnn.GNN((1,) + (args.n_features,) * args.depth, args.n_machines, adj, args.radius)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)


# In[ ]:


with tf.device(args.device):
    x = tf.constant(adj.sum(1), dtype=tf.float32)
    x -= tf.reduce_mean(x)
    x /= tf.sqrt(tf.reduce_mean(tf.square(x)))


# In[ ]:


global_step = tf.train.get_or_create_global_step()
with tf.device(args.device):
    for i in range(args.n_iterations):
        global_step.assign_add(1)
        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            with eager.GradientTape() as tape:
                z = gnn(x, training=True)
                rslt, r, b_node, b_edge, entropy = objective(z, training=True)
            gradients = tape.gradient(rslt, gnn.variables)
            optimizer.apply_gradients(zip(gradients, gnn.variables), global_step=tf.train.get_or_create_global_step())

            r, b_node, b_edge, entropy = objective(z)
            print('[iteration %d]%f %f %f %f %f' % (i + 1, rslt, r, b_node, b_edge, entropy))
            tf.contrib.summary.scalar('objective', rslt)
            tf.contrib.summary.scalar('replication-factor', r)
            tf.contrib.summary.scalar('node-balancedness', b_node)
            tf.contrib.summary.scalar('edge-balancedness', b_edge)
            tf.contrib.summary.scalar('entropy', entropy)


# In[ ]:


# ckpt = eager.Checkpoint(model=gnn)
# ckpt.save('./models/gnn')


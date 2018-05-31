
# coding: utf-8

# In[ ]:


import argparse
import functools
import networkx as nx
import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow.contrib.eager as eager
import my


# In[ ]:


'''
args = argparse.Namespace()
args.depth = 10
args.device = '/cpu:0'
# args.device = '/device:GPU:0' # TODO
args.graph = 'soc-Epinions1'
# args.graph = 'soc-Slashdot0811'
# args.graph = 'soc-Slashdot0902'
args.lambda_node = 1
args.lambda_edge = 1
args.lambda_entropy = 1
args.n_features = 16
args.n_iterations = 500
args.n_machines = 10
args.radius = 3
'''

parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, default=10)
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


class Objective:
    def __init__(self, adj, n_machines, lambda_node, lambda_edge, lambda_entropy):
        adj = sp.sparse.triu(adj)
        self.n_nodes, self.n_edges = adj.shape[0], len(adj.data)
        adj = my.sparse_sp2tf(adj)
        adj = tf.cast(adj, tf.float32)
        e_idx = tf.expand_dims(tf.range(0, self.n_edges, dtype=tf.int64), 1)
        idx0 = tf.concat((tf.expand_dims(adj.indices[:, 0], 1), e_idx), 1)
        idx1 = tf.concat((tf.expand_dims(adj.indices[:, 1], 1), e_idx), 1)
        idx = tf.concat((idx0, idx1), 0)
        self.s = tf.SparseTensor(idx, tf.ones(idx.shape[0]), dense_shape=(self.n_nodes, self.n_edges))
        self.n_machines = n_machines
        self.lambda_node, self.lambda_edge, self.lambda_entropy = lambda_node, lambda_edge, lambda_entropy

    def __call__(self, x, training=False):
        # TODO incorporate extent of concentration in objective
        # TODO return expected number of edges per machine

        p = tf.nn.softmax(x)
        log_p = tf.nn.log_softmax(x)
        entropy = -tf.reduce_sum(p * log_p) / float(self.n_edges)
        
        if training:
            y = tf.multinomial(p, num_samples=1)
            y = tf.squeeze(y)
            y = tf.one_hot(y, self.n_machines)
        else:
            y = p
        
        z = tf.sparse_tensor_dense_matmul(self.s, y)
        z = tf.minimum(z, 1)

        sum_z = tf.reduce_sum(z)
        r = sum_z / float(self.n_nodes)
        
        q_node = (tf.reduce_sum(z, 0) + 1) / sum_z
        b_node = tf.reduce_sum(q_node * tf.log(q_node))
        
        q_edge = (tf.reduce_sum(y, 0) + 1) / tf.reduce_sum(y)
        b_edge = tf.reduce_sum(q_edge * tf.log(q_node))

        if training:
#             joint = r
            joint = r + self.lambda_node * b_node + self.lambda_edge * b_edge + self.lambda_entropy * entropy
            objective = joint * tf.reduce_sum(y * log_p) / float(self.n_edges)
            return objective, r, b_node, b_edge, entropy
        else:
            return r, b_node, b_edge, entropy


# In[ ]:


class GNNModule(tf.keras.Model):
    def __init__(self, in_features, out_features, adj, deg, radius, activation):
        super().__init__()
        self.adj, self.deg, self.radius = adj, deg, radius
        new_dense = lambda: tf.keras.layers.Dense(input_shape=(in_features,), units=out_features, use_bias=False)
        self.alpha1, self.alpha2, self.alpha3 = new_dense(), new_dense(), new_dense()
        for i in range(radius):
            setattr(self, 'alpha%d' % (i + 4), new_dense())
        self.beta1, self.beta2, self.beta3 = new_dense(), new_dense(), new_dense()
        for i in range(radius):
            setattr(self, 'beta%d' % (i + 4), new_dense())
        self.bn_alpha, self.bn_beta = tf.keras.layers.BatchNormalization(axis=1), tf.keras.layers.BatchNormalization(axis=1)
        self.activation = activation
    
    def call(self, x, training=False):
        deg = self.deg * x
        u = tf.reduce_mean(x, 1, keepdims=True) + tf.zeros_like(x)
        adj = [tf.sparse_tensor_dense_matmul(self.adj, x)]
        matmul = lambda x, y: tf.sparse_tensor_dense_matmul(y, x)
        for i in range(self.radius - 1):
            adj.append(functools.reduce(matmul, (self.adj,) * 2 ** i, adj[-1]))
        alpha = self.alpha1(x) + self.alpha2(x) + self.alpha3(x) +             sum(getattr(self, 'alpha%d' % (i + 4))(a) for i, a in enumerate(adj))
        alpha = self.bn_alpha(self.activation(alpha), training=training)
        beta = self.beta1(x) + self.beta2(x) + self.beta3(x) +             sum(getattr(self, 'beta%d' % (i + 4))(a) for i, a in enumerate(adj))
        beta = self.bn_beta(beta, training=training)
        return tf.concat((alpha, beta), 1)

class EdgeDense(tf.keras.Model):
    def __init__(self, in_features, out_features, adj):
        super().__init__()
        adj = sp.sparse.triu(adj)
        self.adj = tf.cast(my.sparse_sp2tf(adj), tf.float32)
        self.out_features = out_features
        for i in range(out_features):
            setattr(self, 'dense%d' % i, tf.keras.layers.Dense(input_shape=(in_features,), units=1))
    
    def call(self, x):
        z_list = []
        for i in range(self.out_features):
            z = getattr(self, 'dense%d' % i)(x)
            u = z * self.adj
            v = tf.transpose(z) * self.adj
            z = u.values + v.values
            z_list.append(tf.reshape(z, (-1, 1)))
        z = tf.concat(z_list, 1)
        return z

class GNN(tf.keras.Model):
    def __init__(self, features, n_machines, adj, radius, activation=tf.keras.activations.relu):
        super().__init__()
        self.dense = EdgeDense(features[-1], n_machines, adj)
        deg = tf.constant(adj.sum(1), dtype=tf.float32)
        adj = tf.cast(my.sparse_sp2tf(adj), tf.float32)
        for i, (m, n) in enumerate(zip(features[:-1], features[1:])):
            setattr(self, 'layer%d' % i, GNNModule(m, n, adj, deg, radius, activation))
        self.n_layers = i + 1
    
    def call(self, x, training=False):
        for i in range(self.n_layers):
            x = getattr(self, 'layer%d' % i)(x, training)
        x = self.dense(x)
        return x


# In[ ]:


g = my.read_edgelist('data/' + args.graph)
adj = nx.adj_matrix(g)
# x = sp.sparse.random(1000, 1000, 1e-5, data_rvs=lambda shape: np.ones(shape))
# adj = (x + x.transpose()).minimum(1)


# In[ ]:


with tf.device(args.device):
    objective = Objective(adj, args.n_machines, args.lambda_node, args.lambda_edge, args.lambda_entropy)
    gnn = GNN((1,) + (args.n_features,) * args.depth, args.n_machines, adj, args.radius)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)


# In[ ]:


with tf.device(args.device):
    x = tf.constant(adj.sum(1), dtype=tf.float32)
    x -= tf.reduce_mean(x)
    x /= tf.sqrt(tf.reduce_mean(tf.square(x)))


# In[ ]:


global_step=tf.train.get_or_create_global_step()
with tf.device(args.device):
    for i in range(args.n_iterations):
        global_step.assign_add(1)
        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            with eager.GradientTape() as tape:
                z = gnn(x, training=True)
                rslt, r, b_node, b_edge, entropy = objective(z, training=True)
            gradients = tape.gradient(rslt, gnn.variables)
            optimizer.apply_gradients(zip(gradients, gnn.variables)) # TODO tf.train.get_or_create_global_step

            r, b_node, b_edge, entropy = objective(z)
            print('[iteration %d]%f %f %f %f %f' % (i + 1, rslt, r, b_node, b_edge, entropy))
            tf.contrib.summary.scalar('objective', rslt)
            tf.contrib.summary.scalar('replication-factor', r)
            tf.contrib.summary.scalar('node-balancedness', b_node)
            tf.contrib.summary.scalar('edge-balancedness', b_edge)
            tf.contrib.summary.scalar('entropy', entropy)



# coding: utf-8

# In[1]:


import argparse
import networkx as nx
import tensorflow as tf # TODO use gpu
import tensorflow.contrib.eager as eager
tf.enable_eager_execution()
import my


# In[2]:


args = argparse.Namespace()
args.depth = 10
# args.device = '/cpu:0'
args.device = '/device:GPU:0'
args.graph = 'soc-Epinions1'
args.n_features = 8
args.n_machines = 10
args.radius = 3


# In[3]:


class Objective:
    def __init__(self, adj, n_machines):
        n_nodes, n_edges = adj.dense_shape[0], adj.indices.shape[0]
        e_idx = tf.expand_dims(tf.range(0, n_edges, dtype=tf.int64), 1)
        idx0 = tf.concat((tf.expand_dims(adj.indices[:, 0], 1), e_idx), 1)
        idx1 = tf.concat((tf.expand_dims(adj.indices[:, 1], 1), e_idx), 1)
        idx = tf.concat((idx0, idx1), 0)
        self.s = tf.SparseTensor(idx, tf.ones(idx.shape[0]), dense_shape=(n_nodes, n_edges))
        self.n_machines = n_machines

    def __call__(self, x):
        y = tf.multinomial(x, num_samples=1)
        y = tf.squeeze(y)
        y = tf.one_hot(y, self.n_machines)
        z = tf.sparse_tensor_dense_matmul(self.s, y)
        # TODO capping
        r = tf.reduce_sum(z)
        p = (tf.reduce_sum(z, 1) + 1) / r
        b = -tf.reduce_sum(p * tf.log(p))
        objective = -tf.reduce_sum((r + b) * y * tf.log(x))
        return objective


# In[5]:


class GNNModule(tf.keras.Model):
    def __init__(self, in_features, out_features, adj, radius, activation):
        super().__init__()
        self.adj, self.radius = adj, radius
        self.deg = tf.sparse_reduce_sum(adj[0], 1, keep_dims=True)
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
        self.adj = adj
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
#             z = tf.sparse_add(z * self.adj, tf.transpose(z) * self.adj).values
            z_list.append(tf.reshape(z, (-1, 1)))
        z = tf.concat(z_list, 1)
        return z

class GNN(tf.keras.Model):
    def __init__(self, features, n_machines, adj, radius, activation=tf.keras.activations.relu):
        super().__init__()
        for i, (m, n) in enumerate(zip(features[:-1], features[1:])):
            setattr(self, 'layer%d' % i, GNNModule(m, n, adj, radius, activation))
        self.n_layers = i + 1
        self.dense = EdgeDense(features[-1], n_machines, adj[0])
    
    def call(self, x, training=False):
        for i in range(self.n_layers):
            x = getattr(self, 'layer%d' % i)(x, training)
        x = self.dense(x)
        x = tf.exp(x) / tf.reduce_sum(tf.exp(x), 1, keepdims=True)
        return x


# In[6]:


g = my.read_edgelist(args.graph)


# In[8]:


with tf.device(args.device):
    adj = tf.cast(my.sparse_sp2tf(nx.adj_matrix(g)), tf.float32)
    objective = Objective(adj_list[0], args.n_machines)
    gnn = GNN((1,) + (args.n_features,) * args.depth, args.n_machines, adj_list, args.radius)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)


# In[ ]:


with tf.device(args.device):
    for i in range(100):
        x = tf.sparse_reduce_sum(adj_list[0], 1, keep_dims=True)
        x -= tf.reduce_mean(x)
        x /= tf.sqrt(tf.reduce_mean(tf.square(x)))
        with eager.GradientTape() as tape:
            x = gnn(x, training=True)
            x = objective(x)
            print(x)
        gradients = tape.gradient(x, gnn.variables)
        optimizer.apply_gradients(zip(gradients, gnn.variables)) # TODO tf.train.get_or_create_global_step


# In[11]:


with tf.device(args.device):
    x = tf.ones(3)


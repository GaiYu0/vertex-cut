import functools
import networkx as nx
import scipy as sp
import tensorflow as tf
import my


class GNNModule(tf.keras.Model):
    def __init__(self, in_features, out_features, adj, deg, radius, activation):
        super().__init__()
        self.adj, self.deg, self.radius = adj, deg, radius

        new_dense = lambda: tf.keras.layers.Dense(input_shape=(in_features,),
                                                  units=out_features, use_bias=False)

        self.alpha1, self.alpha2, self.alpha3 = new_dense(), new_dense(), new_dense()
        for i in range(radius):
            setattr(self, 'alpha%d' % (i + 4), new_dense())

        self.beta1, self.beta2, self.beta3 = new_dense(), new_dense(), new_dense()
        for i in range(radius):
            setattr(self, 'beta%d' % (i + 4), new_dense())

        self.bn_alpha = tf.keras.layers.BatchNormalization(axis=1)
        self.bn_beta = tf.keras.layers.BatchNormalization(axis=1)

        self.activation = activation
    
    def call(self, x, training=False):
        deg = self.deg * x
        u = tf.reduce_mean(x, 1, keepdims=True) + tf.zeros_like(x)
        adj = [tf.sparse_tensor_dense_matmul(self.adj, x)]

        matmul = lambda x, y: tf.sparse_tensor_dense_matmul(y, x)
        for i in range(self.radius - 1):
            adj.append(functools.reduce(matmul, (self.adj,) * 2 ** i, adj[-1]))

        alpha = self.alpha1(x) + self.alpha2(x) + self.alpha3(x) + \
            sum(getattr(self, 'alpha%d' % (i + 4))(a) for i, a in enumerate(adj))
        alpha = self.bn_alpha(self.activation(alpha), training=training)

        beta = self.beta1(x) + self.beta2(x) + self.beta3(x) + \
            sum(getattr(self, 'beta%d' % (i + 4))(a) for i, a in enumerate(adj))
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
    def __init__(self, g, features, n_machines, radius, activation):
        super().__init__()
        adj = nx.adj_matrix(g)
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

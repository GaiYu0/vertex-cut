# TODO


import functools
import networkx as nx
import scipy as sp
import tensorflow as tf
import my


class GNNLayer(tf.keras.Model):
    def __init__(self, in_features, out_features,
                 adj, deg, adj_lg, deg_lg, p, radius, activation):
        super().__init__()

        new_dense = lambda: tf.keras.layers.Dense(input_shape=(in_features,),
                                                  units=out_features, use_bias=False)

        self.alpha1, self.alpha2, self.alpha3 = \
            new_dense(), new_dense(), new_dense()

        self.alpha_lg1, self.alpha_lg2, self.alpha_lg3 = \
            new_dense(), new_dense(), new_dense()

        for i in range(radius):
            setattr(self, 'alpha%d' % (i + 4), new_dense())
            setattr(self, 'alpha_lg%d' % (i + 4), new_dense())
        setattr(self, 'alpha%d' % (radius + 4), new_dense())
        setattr(self, 'alpha_lg%d' % (radius + 4), new_dense())

        self.beta1, self.beta2, self.beta3 = \
            new_dense(), new_dense(), new_dense()

        self.beta_lg1, self.beta_lg2, self.beta_lg3 = \
            new_dense(), new_dense(), new_dense()

        for i in range(radius):
            setattr(self, 'beta%d' % (i + 4), new_dense())
            setattr(self, 'beta_lg%d' % (i + 4), new_dense())
        setattr(self, 'beta%d' % (radius + 4), new_dense())
        setattr(self, 'beta_lg%d' % (radius + 4), new_dense())

        self.bn_alpha = tf.keras.layers.BatchNormalization(axis=1)
        self.bn_beta = tf.keras.layers.BatchNormalization(axis=1)

        self.bn_alpha_lg = tf.keras.layers.BatchNormalization(axis=1)
        self.bn_beta_lg = tf.keras.layers.BatchNormalization(axis=1)

        self.adj, self.deg = adj, deg
        self.adj_lg, self.deg_lg = adj_lg, deg_lg
        self.p = p

        self.radius = radius
        self.activation = activation
    
    def call(self, x, y, training=False):
        deg = self.deg * x
        u = tf.reduce_mean(x, 1, keepdims=True) + tf.zeros_like(x)
        adj = [tf.sparse_tensor_dense_matmul(self.adj, x)]

        deg_lg = self.deg_lg * y
        u_lg = tf.reduce_mean(y, 1, keepdims=True) + tf.zeros_like(y)
        adj_lg = [tf.sparse_tensor_dense_matmul(self.adj_lg, y)]

        py = tf.sparse_tensor_dense_matmul(self.p, y)
        px = tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.p), x)

        matmul = lambda x, y: tf.sparse_tensor_dense_matmul(y, x)
        for i in range(self.radius - 1):
            adj.append(functools.reduce(matmul, (self.adj,) * 2 ** i, adj[-1]))
            adj_lg.append(functools.reduce(matmul, (self.adj_lg,) * 2 ** i, adj_lg[-1]))

        alpha = self.alpha1(x) + self.alpha2(x) + self.alpha3(x) + \
            sum(getattr(self, 'alpha%d' % (i + 4))(a) for i, a in enumerate(adj)) + \
            getattr(self, 'alpha%d' % (self.radius + 4))(py)
        alpha = self.bn_alpha(self.activation(alpha), training=training)

        beta = self.beta1(x) + self.beta2(x) + self.beta3(x) + \
            sum(getattr(self, 'beta%d' % (i + 4))(a) for i, a in enumerate(adj)) + \
            getattr(self, 'beta%d' % (self.radius + 4))(py)
        beta = self.bn_beta(beta, training=training)

        alpha_lg = self.alpha_lg1(y) + self.alpha_lg2(y) + self.alpha_lg3(y) + \
            sum(getattr(self, 'alpha_lg%d' % (i + 4))(a) for i, a in enumerate(adj_lg)) + \
            getattr(self, 'alpha%d' % (self.radius + 4))(px)
        alpha_lg = self.bn_alpha_lg(self.activation(alpha_lg), training=training)

        beta_lg = self.beta_lg1(y) + self.beta_lg2(y) + self.beta_lg3(y) + \
            sum(getattr(self, 'beta_lg%d' % (i + 4))(a) for i, a in enumerate(adj_lg)) + \
            getattr(self, 'beta%d' % (self.radius + 4))(px)
        beta_lg = self.bn_beta_lg(beta_lg, training=training)

        return tf.concat((alpha, beta), 1), tf.concat((alpha_lg, beta_lg), 1)


class GNN(tf.keras.Model):
    def __init__(self, g, features, n_machines, radius, activation, device):
        super().__init__()
        adj = nx.adj_matrix(g)
        p = my.adj2p(sp.sparse.triu(adj))
        adj = tf.cast(my.sparse_sp2tf(adj), tf.float32)
        deg = tf.expand_dims(tf.sparse_reduce_sum(adj, 1), 1)

        lg = nx.line_graph(g)
        adj_lg = tf.cast(my.sparse_sp2tf(nx.adj_matrix(lg)), tf.float32)
        deg_lg = tf.expand_dims(tf.sparse_reduce_sum(adj_lg, 1), 1)

        for i, (m, n) in enumerate(zip(features[:-1], features[1:])):
            setattr(self, 'layer%d' % i,
                    GNNLayer(m, n, adj, deg, adj_lg, deg_lg, p, radius, activation))

        self.n_layers = i + 1

        self.dense = tf.keras.layers.Dense(input_shape=(n,), units=n_machines)

        self.device = device

        with tf.device(device):
            x = deg
            x -= tf.reduce_mean(x)
            x /= tf.sqrt(tf.reduce_mean(tf.square(x)))

            y = deg_lg
            y -= tf.reduce_mean(y)
            y /= tf.sqrt(tf.reduce_mean(tf.square(y)))

            self.x, self.y = x, y

    def call(self, placeholder, training=False):
        with tf.device(self.device):
            x, y = self.x, self.y
            for i in range(self.n_layers):
                x, y = getattr(self, 'layer%d' % i)(x, y, training)

            return self.dense(y)

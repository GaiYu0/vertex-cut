import scipy as sp
import tensorflow as tf
import my


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

    def __call__(self, x, training=False, probability=False):
        # TODO return expected number of edges per machine

        if probability:
            p = x
            log_p = tf.log(p + 1e-10)
        else:
            p = tf.nn.softmax(x)
            log_p = tf.nn.log_softmax(x)

        entropy = -tf.reduce_sum(p * log_p) / float(self.n_edges)
        
        if training:
            y = tf.multinomial(log_p, num_samples=1)
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
            # TODO sampling multiple times
            # TODO reward = overlapping between oracle and gnn; start from cross-entropy
            joint = r + self.lambda_node * b_node + self.lambda_edge * b_edge + self.lambda_entropy * entropy
            objective = joint * tf.reduce_sum(y * log_p) / float(self.n_edges)
            return objective, r, b_node, b_edge, entropy
        else:
            return r, b_node, b_edge, entropy

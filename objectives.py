import networkx as nx
import scipy as sp
import tensorflow as tf
import my


class Objective:
    def __init__(self,
                 g, lambda_node, lambda_edge, lambda_entropy, n_machines, n_samples):
        adj = sp.sparse.triu(nx.adj_matrix(g))
        self.n_nodes, self.n_edges = adj.shape[0], len(adj.data)

        self.p = my.adj2p(adj)

        self.lambda_node = lambda_node
        self.lambda_edge = lambda_edge
        self.lambda_entropy = lambda_entropy
        self.n_machines = n_machines
        self.n_samples = n_samples

    def train(self, p, log_p):
        entropy = -tf.stop_gradient(tf.reduce_sum(p * log_p)) / float(self.n_edges)

        objective = 0
        for i in range(self.n_samples):
            y = tf.multinomial(log_p, num_samples=1)
            y = tf.one_hot(tf.squeeze(y), self.n_machines)

            z = tf.minimum(1, tf.sparse_tensor_dense_matmul(self.p, y))

            sum_z = tf.reduce_sum(z)
            r = sum_z / float(self.n_nodes)
            
            q_node = (tf.reduce_sum(z, 0) + 1) / sum_z
            b_node = -tf.reduce_sum(q_node * tf.log(q_node))
            
            q_edge = (tf.reduce_sum(y, 0) + 1) / tf.reduce_sum(y)
            b_edge = -tf.reduce_sum(q_edge * tf.log(q_node))

            joint = r - self.lambda_node * b_node \
                      - self.lambda_edge * b_edge \
                      + self.lambda_entropy * entropy

            objective += joint * tf.reduce_sum(y * log_p) / float(self.n_edges)

        return objective / float(self.n_samples)
         
    def test(self, p, log_p):
        entropy = -tf.stop_gradient(tf.reduce_sum(p * log_p)) / float(self.n_edges)

        r, b_node, b_edge = 0, 0, 0
        for i in range(self.n_samples):
            y = tf.multinomial(log_p, num_samples=1)
            y = tf.one_hot(tf.squeeze(y), self.n_machines)

            z = tf.minimum(1, tf.sparse_tensor_dense_matmul(self.p, y))

            sum_z = tf.reduce_sum(z)
            r += sum_z / float(self.n_nodes)
            
            q_node = (tf.reduce_sum(z, 0) + 1) / sum_z
            b_node += -tf.reduce_sum(q_node * tf.log(q_node))
            
            q_edge = (tf.reduce_sum(y, 0) + 1) / tf.reduce_sum(y)
            b_edge += -tf.reduce_sum(q_edge * tf.log(q_node))

        r /= float(self.n_samples)
        b_node /= float(self.n_samples)
        b_edge /= float(self.n_samples)

        objective = r - self.lambda_node * b_node \
                      - self.lambda_edge * b_edge \
                      + self.lambda_entropy * entropy

        return objective, r, b_node, b_edge, entropy

    def __call__(self, x, probability=False, oracle=None, training=False):
        if probability:
            p = x
            log_p = tf.log(p + 1e-10)
        else:
            p = tf.nn.softmax(x)
            log_p = tf.nn.log_softmax(x)

        if training:
            return self.train(p, log_p)
        else:
            return self.test(p, log_p)

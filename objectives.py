import networkx as nx
import numpy as np
import scipy as sp
import tensorflow as tf
import my


class Objective:
    def __init__(self, g, n_machines, lambda_node, lambda_edge, lambda_entropy):
        self.adj = sp.sparse.triu(nx.adj_matrix(g))
        self.p = my.adj2p(self.adj)

        self.n_nodes, self.n_edges = self.adj.shape[0], len(self.adj.data)

        self.lambda_node = lambda_node
        self.lambda_edge = lambda_edge
        self.lambda_entropy = lambda_entropy
        self.n_machines = n_machines

    def __call__(self, x, n_samples, probability=False, training=False):
        if probability:
            p = x
            log_p = tf.log(p + 1e-10)
        else:
            p = tf.nn.softmax(x)
            log_p = tf.nn.log_softmax(x)

        if training:
            return self.train(p, log_p, n_samples)
        else:
            return self.test(p, log_p, n_samples)

    @staticmethod
    def entropy(p):
        return -tf.reduce_sum(p * tf.log(p))

    def sample(self, log_p):
        e_asgmnt = tf.multinomial(log_p, num_samples=1)
        e_asgmnt = tf.one_hot(tf.squeeze(e_asgmnt), self.n_machines)
        n_asgmnt = tf.minimum(1, tf.sparse_tensor_dense_matmul(self.p, e_asgmnt))
        return n_asgmnt, e_asgmnt

    def scores(self, n_asgmnt, e_asgmnt):
        z = tf.reduce_sum(n_asgmnt)
        r = z / float(self.n_nodes)

        q_node = (tf.reduce_sum(n_asgmnt, 0) + 1) / z
        b_node = self.entropy(q_node)

        q_edge = (tf.reduce_sum(e_asgmnt, 0) + 1) / tf.reduce_sum(e_asgmnt)
        b_edge = self.entropy(q_edge)

        return r, b_node, b_edge


    def test(self, p, log_p, n_samples):
        entropy = tf.reduce_sum(p * log_p) / float(self.n_edges)

        asgmnt_list = [self.sample(log_p) for i in range(n_samples)]
        scores = [self.scores(n_asgmnt, e_asgmnt) for n_asgmnt, e_asgmnt in asgmnt_list]

        r, b_node, b_edge = zip(*scores)
        r = sum(r) / float(n_samples)
        b_node = sum(b_node) / float(n_samples)
        b_edge = sum(b_edge) / float(n_samples)

        objective = r - self.lambda_node    * b_node \
                      - self.lambda_edge    * b_edge \
                      + self.lambda_entropy * entropy

        return objective, r, b_node, b_edge, entropy



class OracleObjective(Objective):
    def __init__(self, g, partitioner, *args):
        super().__init__(g, *args)
        ptn = partitioner(self.adj, self.n_machines, np.random.RandomState())
        ptn.partition()
        self.n_asgmnt = tf.constant(ptn.vassign, dtype=tf.float32)
        self.e_asgmnt = tf.one_hot(tf.constant(ptn.eassign), self.n_machines)


class CrossEntropy(OracleObjective):
    def train(self, p, log_p, _):
        return -tf.reduce_sum(self.e_asgmnt * log_p) / float(self.n_edges)


class OraclePolicyGradient(OracleObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.node = kwargs['node']

    def train(self, p, log_p, n_samples):
        entropy = self.lambda_entropy * tf.reduce_sum(p * log_p) / float(self.n_edges)

        asgmnt_list = [self.sample(log_p) for i in range(n_samples)]
        if self.node:
            commonz = [tf.reduce_sum(n_asgmnt * self.n_asgmnt) / float(self.n_nodes)
                           for n_asgmnt, _ in asgmnt_list]
        else:
            commonz = [tf.reduce_sum(e_asgmnt * self.e_asgmnt) / float(self.n_edges)
                           for _, e_asgmnt in asgmnt_list]
        log_pz = [tf.reduce_sum(e_asgmnt * log_p) / float(self.n_edges)
                 for _, e_asgmnt in asgmnt_list]
        mean = sum(common * log_p for common, log_p in zip(commonz, log_pz)) / n_samples

        return mean + self.lambda_entropy * entropy


class PolicyGradient(Objective):
    def __init__(self, *args):
        super().__init__(*args)

    def train(self, p, log_p, n_samples):
        entropy = self.lambda_entropy * tf.reduce_sum(p * log_p) / float(self.n_edges)

        asgmnt_list = [self.sample(log_p) for i in range(n_samples)]
        scores = [self.scores(n_asgmnt, e_asgmnt) for n_asgmnt, e_asgmnt in asgmnt_list]

        rz, b_nodez, b_edgez = zip(*scores)
        log_pz = [tf.reduce_sum(e_asgmnt * log_p) / float(self.n_edges)
                 for _, e_asgmnt in asgmnt_list]
        mean = sum((r + self.lambda_node * b_node + self.lambda_edge * b_edge) * log_p
                    for r, b_node, b_edge, log_p in zip(rz, b_nodez, b_edgez, log_pz)
               ) / float(n_samples)

        return mean + self.lambda_entropy * entropy

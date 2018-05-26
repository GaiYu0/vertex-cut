import numpy as np
import torch as th

def sp2th(coo):
    """
    Parameters
    ----------
    coo : scipy.sparse.coo.coo_matrix
    """

    row = np.reshape(coo.row, (1, -1))
    col = np.reshape(coo.col, (1, -1))
    idx = th.from_numpy(np.vstack((row, col))).long()

import os

import numpy as np

from scipy.sparse.linalg import LinearOperator, eigs

def ensure_dir_exists(filename):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)


def cos_max_principal_angle(X, Y):
    # NOTE: Doesn't work if X and Y are rank-deficient.

    d = X.shape[0]

    X = np.reshape(X, (d, -1)).T
    Y = np.reshape(Y, (d, -1)).T

    X, RX = np.linalg.qr(X, 'reduced')
    Y, RY = np.linalg.qr(Y, 'reduced')

    M = X.T @ Y

    _, S, _ = np.linalg.svd(M)

    return S[-1]


def subspace_dist(X, Y):
    p = X.shape[1]

    X = np.linalg.qr(X.T)[0].T
    Y = np.linalg.qr(Y.T)[0].T

    op = lambda v: (v @ X.T) @ X - (v @ Y.T) @ Y
    op_transp = lambda v: op(v.T).T

    v = np.random.randn(p)

    linop = LinearOperator(2 * (p,), matvec=op_transp)

    lam, _ = eigs(linop, k=1, which='LM')

    dist = np.abs(lam[0])

    return dist

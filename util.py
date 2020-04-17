import os

import numpy as np


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

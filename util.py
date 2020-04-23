import os

import numpy as np

from scipy.linalg import subspace_angles

def ensure_dir_exists(filename):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)


def write_gplt_binary_matrix(filename, A, lab1=None, lab2=None):
    if lab1 is None:
        lab1 = np.arange(1, A.shape[0] + 1)

    if lab2 is None:
        lab2 = np.arange(1, A.shape[1] + 1)

    M = np.zeros(tuple(sz + 1 for sz in A.shape))

    M[0, 0] = A.shape[1]
    M[1:, 0] = lab2
    M[0, 1:] = lab1
    M[1:, 1:] = np.flipud(A).T

    with open(filename, 'wb') as f:
        f.write(M.T.astype(np.float32).tobytes())


def cos_max_principal_angle(X, Y):
    theta = np.max(subspace_angles(X.T, Y.T))

    return np.cos(theta)


def subspace_dist(X, Y):
    theta = np.max(subspace_angles(X.T, Y.T))

    return np.sin(theta)

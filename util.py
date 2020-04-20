import os

import numpy as np

from scipy.linalg import subspace_angles

def ensure_dir_exists(filename):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)


def cos_max_principal_angle(X, Y):
    theta = np.max(subspace_angles(X.T, Y.T))

    return np.cos(theta)


def subspace_dist(X, Y):
    theta = np.max(subspace_angles(X.T, Y.T))

    return np.sin(theta)

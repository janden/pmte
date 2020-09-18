import os

import numpy as np

from scipy.ndimage import rotate


def _root_dir():
    dirname = os.path.dirname(os.path.abspath(__file__))
    return os.path.split(dirname)[0]


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
        f.write(M.astype(np.float32).tobytes('F'))


def load_float32(fname):
    with open(fname, 'rb') as f:
        buf = f.read()
    return np.frombuffer(buf, dtype=np.float32)


def load_sim_images(n=1000):
    filename = os.path.join(_root_dir(), 'data', 'signal.npz')

    f = np.load(filename)

    ims0 = f['ims']
    ims0 = ims0.astype(np.float32) / 255

    n0 = ims0.shape[0]
    sz0 = ims0.shape[-2:]

    rng = np.random.default_rng(0)

    idx = rng.choice(n0, n)
    ims = ims0[idx]

    angles = rng.uniform(0, 360, n)

    for im, angle in zip(ims, angles):
        rotate(im, angle, reshape=False, output=im)

    return ims


def load_exp_images(n=120):
    filename = os.path.join(_root_dir(), 'data', 'exp.npz')

    f = np.load(filename)

    ims0 = f['ims']
    projs0 = f['projs']

    ims_range = f['ims_range']
    projs_range = f['projs_range']

    n0 = ims0.shape[0]

    if n > n0:
        raise RuntimeError('Number of images n must be smaller than %d.' % n0)

    ims0 = ims0.astype(np.float32) / 255
    projs0 = projs0.astype(np.float32) / 255

    ims0 = ims0 * (ims_range[1] - ims_range[0]) + ims_range[0]
    projs0 = projs0 * (projs_range[1] - projs_range[0]) + projs_range[0]

    ims = ims0[:n]
    projs = projs0[:n]

    return ims, projs

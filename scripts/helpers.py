import os

import numpy as np
import json

from scipy.ndimage import rotate

import matplotlib.pyplot as plt


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


def save_image(name, image):
    filename = os.path.join(_root_dir(), 'data', name + '.bin')

    ensure_dir_exists(filename)

    write_gplt_binary_matrix(filename, image)


def save_table(name, *args):
    filename = os.path.join(_root_dir(), 'data', name + '.csv')

    n_cols = len(args)

    ensure_dir_exists(filename)

    fmt = ' '.join(('%.15g',) * n_cols)

    with open(filename, 'w') as f:
        for vals in zip(*args):
            line = fmt % vals
            f.write(line + '\n')


def save_dictionary(name, values):
    filename = os.path.join(_root_dir(), 'data', name + '.json')

    ensure_dir_exists(filename)

    with open(filename, 'w') as f:
        json.dump(values, f)
        f.write('\n')


def plot_grayscale_image(im, diverging=True, axis='q', reverse=True):
    if diverging:
        mx = np.max(np.abs(im))
        mn = -mx
    else:
        mn, mx = np.min(im), np.max(im)

    cmap = 'gray'

    if reverse:
        cmap += '_r'

    plt.imshow(im, cmap=cmap, vmin=mn, vmax=mx)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$%s_1$' % (axis,))
    plt.ylabel('$%s_2$' % (axis,))

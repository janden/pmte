import os
import gzip

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
        f.write(M.astype(np.float32).tobytes('F'))


def cos_max_principal_angle(X, Y):
    theta = np.max(subspace_angles(X.T, Y.T))

    return np.cos(theta)


def subspace_dist(X, Y):
    theta = np.max(subspace_angles(X.T, Y.T))

    return np.sin(theta)


def load_float32(fname):
    with open(fname, 'rb') as f:
        buf = f.read()
    return np.frombuffer(buf, dtype=np.float32)


def centered_fftn(x, d):
    x = np.fft.ifftshift(x, axes=range(-d, 0))
    x = np.fft.fftn(x, axes=range(-d, 0))
    x = np.fft.fftshift(x, axes=range(-d, 0))

    return x


def centered_ifftn(x, d):
    x = np.fft.ifftshift(x, axes=range(-d, 0))
    x = np.fft.ifftn(x, axes=range(-d, 0))
    x = np.fft.fftshift(x, axes=range(-d, 0))

    return x


def load_sim_images(n=1000):
    filename = 'signal.bin'
    dtype = np.uint8
    shape = (1000, 128, 128)

    if n > shape[0]:
        raise RuntimeError('n must be smaller than %d.' % shape[0])

    with open(filename, 'rb') as f:
        buf = f.read()
    buf = gzip.decompress(buf)

    im = np.frombuffer(buf, dtype=dtype)
    im = im.reshape(shape)
    im = im.astype(np.float32) / 255

    im = im[:n]

    return im

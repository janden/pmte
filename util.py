import os
import gzip

import numpy as np

from scipy.linalg import subspace_angles

from scipy.ndimage import rotate

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


def load_new_sim_images(n=1000):
    filename = 'signal.npz'

    f = np.load(filename)

    ims0 = f['ims']
    ims0 = ims0.astype(np.float32)/ 255

    n0 = ims0.shape[0]
    sz0 = ims0.shape[-2:]

    rng = np.random.default_rng(0)

    idx = rng.choice(n0, n)
    ims = ims0[idx]

    angles = rng.uniform(0, 360, n)

    for im, angle in zip(ims, angles):
        rotate(im, angle, reshape=False, output=im)

    return ims


def load_exp_images(n=1024):
    filename = 'first1024.mat'

    from scipy.io import loadmat

    f = loadmat(filename)

    im = f['im'].T
    proj = f['proj'].T
    groups = f['groups'][0]

    if n > im.shape[0]:
        raise RuntimeError('n must be smaller than %d.' % im.shape[0])

    im = im[:n]
    proj = proj[:n]
    groups = groups[:n]

    return im, proj, groups


def load_new_exp_images(n=120):
    filename = 'exp.npz'

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


def log_slope(x, y):
    x = np.log(x)
    y = np.log(y)

    x = x - np.mean(x)
    y = y - np.mean(y)

    beta = np.dot(x, y) / np.dot(x, x)

    return beta


def grid(sz):
    sz = np.array(sz)

    if sz.ndim == 0:
        sz = sz[np.newaxis]

    rngs = [np.ceil(np.arange(-N / 2, N / 2)) for N in sz]

    grid = np.meshgrid(*rngs, indexing='ij')

    return grid


def target_win(Nf, W, shifted=False):
    fX, fY = grid((Nf, Nf))
    fX, fY = fX / Nf, fY / Nf

    if not shifted:
        fX = np.fft.ifftshift(fX)
        fY = np.fft.ifftshift(fY)

    rho = np.zeros((Nf, Nf))

    half_W = W / 2

    rho[(np.abs(fX) < half_W) & (np.abs(fY) < half_W)] = 1 / W ** 2
    rho[(np.abs(fX) == half_W) & (np.abs(fY) < half_W)] = 1 / W ** 2 / 2
    rho[(np.abs(fX) < half_W) & (np.abs(fY) == half_W)] = 1 / W ** 2 / 2
    rho[(np.abs(fX) == half_W) & (np.abs(fY) == half_W)] = 1 / W ** 2 / 4

    return rho


def disk_mask(N, R):
    x1, x2 = grid((N, N))

    r = np.hypot(x1, x2)

    mask = (r < R)

    return mask


def square_mask(N, R):
    x1, x2 = grid((N, N))

    mask = (-R <= x1) & (x1 < R) & (-R <= x2) & (x2 < R)

    return mask

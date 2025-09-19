import os

import numpy as np
import json

from scipy.ndimage import rotate

import matplotlib.pyplot as plt

from pmte import util

from urllib.request import urlretrieve
from hashlib import md5


def _root_dir():
    dirname = os.path.dirname(os.path.abspath(__file__))
    return os.path.split(dirname)[0]


def _data_dir():
    return os.path.join(_root_dir(), 'data')


def _results_dir():
    return os.path.join(_root_dir(), 'results')


def _ensure_dir_exists(filename):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)


def _write_gplt_binary_matrix(filename, A, lab1=None, lab2=None):
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


def _get_data(key):
    url_root = "https://zenodo.org/record/4742575/files/"

    files = {}
    files['simulation'] = {'filename': 'simulation.npz',
                           'md5': 'a1a63e339b2d9a6cb033a85654fbf1c8'}
    files['experiment'] = {'filename': 'experiment.npz',
                           'md5': 'f1553d4516921d7f11ca34fc5c427b6a'}

    if not key in files:
        raise IndexError('Data key not found.')

    filename = os.path.join(_data_dir(), files[key]['filename'])

    _ensure_dir_exists(filename)

    if not os.path.exists(filename):
        url = url_root + files[key]['filename']
        urlretrieve(url, filename)

    with open(filename, 'rb') as f:
        md5_hash = md5(f.read()).hexdigest()

    if md5_hash != files[key]['md5']:
        raise RuntimeError('Checksum does not match for %s.' %
                           (filename,))

    return filename


def load_sim_images(n=1000):
    filename = _get_data('simulation')

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
    filename = _get_data('experiment')

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
    filename = os.path.join(_results_dir(), name + '.bin')

    _ensure_dir_exists(filename)

    _write_gplt_binary_matrix(filename, image)


def save_table(name, *args):
    filename = os.path.join(_results_dir(), name + '.csv')

    n_cols = len(args)

    _ensure_dir_exists(filename)

    fmt = ' '.join(('%.15g',) * n_cols)

    with open(filename, 'w') as f:
        for vals in zip(*args):
            line = fmt % vals
            f.write(line + '\n')


def save_dictionary(name, values):
    filename = os.path.join(_results_dir(), name + '.json')

    _ensure_dir_exists(filename)

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


def log_slope(x, y):
    x = np.log(x)
    y = np.log(y)

    x = x - np.mean(x)
    y = y - np.mean(y)

    beta = np.dot(x, y) / np.dot(x, x)

    return beta


def target_spectral_window(Nf, W, shifted=False):
    fX, fY = util.grid((Nf, Nf), shifted=shifted)

    rho = np.zeros((Nf, Nf))

    half_W = W / 2

    rho[(np.abs(fX) < half_W) & (np.abs(fY) < half_W)] = 1 / W ** 2
    rho[(np.abs(fX) == half_W) & (np.abs(fY) < half_W)] = 1 / W ** 2 / 2
    rho[(np.abs(fX) < half_W) & (np.abs(fY) == half_W)] = 1 / W ** 2 / 2
    rho[(np.abs(fX) == half_W) & (np.abs(fY) == half_W)] = 1 / W ** 2 / 4

    return rho


def disk_mask(N, R):
    x1, x2 = util.grid((N, N), normalized=False, shifted=True)

    r = np.hypot(x1, x2)

    mask = (r < R)

    return mask


def square_mask(N, R):
    x1, x2 = util.grid((N, N), normalized=False, shifted=True)

    mask = (-R <= x1) & (x1 < R) & (-R <= x2) & (x2 < R)

    return mask


def generate_field(sig_sz, n, psd_fun=None, gen_sig_sz=None, rng=None,
                   real=True, dtype='float64'):
    if psd_fun is None:
        def psd_fun(x, y):
            return np.ones_like(x)

    if gen_sig_sz is None:
        gen_sig_sz = tuple(2 * sz for sz in sig_sz)

    if rng is None:
        rng = np.random.default_rng()

    dtype = np.dtype(dtype)
    if not dtype in [np.dtype('float32'), np.dtype('float64')]:
        raise ValueError('Invalid dtype. Must be `float32` or `float64`.')

    gen_fun = rng.standard_normal

    block_size = 4096

    d = len(sig_sz)

    grid = util.grid(gen_sig_sz)

    raveled_grid = [rng.ravel() for rng in grid]

    density = psd_fun(*raveled_grid)
    density = density.reshape(gen_sig_sz)

    filter_f = np.sqrt(np.maximum(0, density))

    if real:
        complex_dtype = dtype
    elif dtype == np.dtype('float32'):
        complex_dtype = 'complex64'
    elif dtype == np.dtype('float64'):
        complex_dtype = 'complex128'

    x = np.zeros((n,) + sig_sz, dtype=complex_dtype)

    block_count = int(np.ceil(n / block_size))

    for ell in range(block_count):
        n_block = min(n - ell * block_size, block_size)

        gen_block_sz = (n_block,) + gen_sig_sz

        if real:
            w = gen_fun(gen_block_sz, dtype)
        else:
            w = 1 / np.sqrt(2) * (gen_fun(gen_block_sz, dtype)
                                  + 1J * gen_fun(gen_block_sz, dtype))

        wf = np.fft.fftn(w, axes=range(-d, 0))
        xf_block = wf * filter_f
        x_block = np.fft.ifftn(xf_block, axes=range(-d, 0))

        if real:
            x_block = np.real(x_block)

        ixgrid = np.ix_(range(n_block), *(range(sz) for sz in sig_sz))

        x[ell * block_size: ell * block_size + n_block] = x_block[ixgrid]

    return x

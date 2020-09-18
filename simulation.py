import numpy as np

import util


def generate_field(sig_sz, n, psd_fun=None, gen_sig_sz=None, rng=None,
                   real=True):
    if psd_fun is None:
        def psd_fun(x, y):
            return np.ones_like(x)

    if gen_sig_sz is None:
        gen_sig_sz = tuple(2 * sz for sz in sig_sz)

    if rng is None:
        rng = np.random.default_rng()

    gen_fun = rng.standard_normal

    block_size = 4096

    d = len(sig_sz)

    grid = util.grid(gen_sig_sz)

    raveled_grid = [rng.ravel() for rng in grid]

    density = psd_fun(*raveled_grid)
    density = density.reshape(gen_sig_sz)

    filter_f = np.sqrt(np.maximum(0, density))

    if real:
        dtype = 'float64'
    else:
        dtype = 'complex128'

    x = np.zeros((n,) + sig_sz, dtype=dtype)

    block_count = int(np.ceil(n / block_size))

    for ell in range(block_count):
        n_block = min(n - ell * block_size, block_size)

        gen_block_sz = (n_block,) + gen_sig_sz

        if real:
            w = gen_fun(gen_block_sz)
        else:
            w = 1 / np.sqrt(2) * (gen_fun(gen_block_sz)
                                  + 1J * gen_fun(gen_block_sz))

        wf = np.fft.fftn(w, axes=range(-d, 0))
        xf_block = wf * filter_f
        x_block = np.fft.ifftn(xf_block, axes=range(-d, 0))

        if real:
            x_block = np.real(x_block)

        ixgrid = np.ix_(range(n_block), *(range(sz) for sz in sig_sz))

        x[ell * block_size: ell * block_size + n_block] = x_block[ixgrid]

    return x

import numpy as np


def generate_field(sig_sz, n, psd_fun=None, gen_sig_sz=None, gen_fun=None):
    if psd_fun is None:
        psd_fun = lambda x, y: np.ones_like(x)

    if gen_sig_sz is None:
        gen_sig_sz = tuple(2 * sz for sz in sig_sz)

    if gen_fun is None:
        rng = np.random.default_rng()
        gen_fun = rng.standard_normal

    block_size = 4096

    d = len(sig_sz)

    rngs = [np.fft.fftfreq(sz) for sz in gen_sig_sz]
    grids = np.meshgrid(*rngs, indexing='ij')

    filter_f = np.sqrt(np.maximum(0, psd_fun(*grids)))

    x = np.zeros((n,) + sig_sz)

    block_count = int(np.ceil(n / block_size))

    for ell in range(block_count):
        n_block = min(n - ell * block_size, block_size)

        w = gen_fun((n_block,) + gen_sig_sz)

        wf = np.fft.fftn(w, axes=range(-d, 0))
        xf_block = wf * filter_f
        x_block = np.real(np.fft.ifftn(xf_block, axes=range(-d, 0)))

        ixgrid = np.ix_(range(n_block), *(range(sz) for sz in sig_sz))

        x[ell * block_size: ell * block_size + n] = x_block[ixgrid]

    return x

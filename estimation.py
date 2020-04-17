import numpy as np


def concentration_op(mask, W=1/8, use_sinc=False):
    d = mask.ndim

    W = _ensure_W(W, d)

    sig_sz = mask.shape
    sig_len = np.prod(sig_sz)

    if not use_sinc:
        rngs = [np.fft.fftfreq(sz) for sz in sig_sz]
        grids = np.meshgrid(*rngs, indexing='ij')

        freq_mask = np.full(sig_sz, True)

        for ell in range(d):
            freq_mask &= np.abs(grids[ell]) < W[ell]
    else:
        two_sig_sz = tuple(2 * sz for sz in sig_sz)

        rngs = [sz * np.fft.fftfreq(sz) for sz in two_sig_sz]
        grids = np.meshgrid(*rngs, indexing='ij')

        sinc_kernel = np.ones(two_sig_sz)

        for ell in range(d):
            sinc_kernel *= 2 * W[ell] * np.sinc(2 * W[ell] * grids[ell])

        freq_mask = np.fft.fftn(sinc_kernel, axes=range(-d, 0))

    def _apply(x):
        x = np.reshape(x, x.shape[:1] + sig_sz)

        x = x * mask

        if use_sinc:
            y = np.zeros(x.shape[:1] + two_sig_sz)

            ixgrid = np.ix_(range(x.shape[0]), *(range(sz) for sz in sig_sz))
            y[ixgrid] = x

            x = y

        xf = np.fft.fftn(x, axes=range(-d, 0))

        xf = xf * freq_mask

        x = np.fft.ifftn(xf, axes=range(-d, 0))
        x = np.real(x)

        if use_sinc:
            x = x[ixgrid]

        x = x * mask

        x = np.reshape(x, x.shape[:1] + (sig_len,))

        return x

    def _blocked_apply(x):
        block_size = 64

        n = x.shape[0]

        block_count = int(np.ceil(n / block_size))

        y = np.empty_like(x)

        for ell in range(block_count):
            start = ell * block_size
            stop = min((ell + 1) * block_size, n)
            rng = range(start, stop)

            y[rng] = _apply(x[rng])

        return y

    def _reshaped_apply(x):
        singleton = (x.ndim == 1)

        if singleton:
            x = x[np.newaxis, ...]

        x = _blocked_apply(x)

        if singleton:
            x = x[0]

        return x

    return _reshaped_apply


def calc_rand_tapers(mask, W=1/8, p=5, b=3, gen_fun=None, use_sinc=False):
    if gen_fun is None:
        rng = np.random.default_rng()
        gen_fun = rng.standard_normal

    d = mask.ndim

    # TODO: This W does not correspond to the W used in the paper, since this
    # is the half-bandwidth. To get the W in the paper, multiply by 2.
    W = _ensure_W(W, d)

    sig_sz = mask.shape
    sig_len = np.prod(sig_sz)

    K = int(np.ceil(np.prod(2 * W) * np.sum(mask)))

    op = concentration_op(mask, W=W, use_sinc=use_sinc)

    X = gen_fun((K + p, sig_len))

    # TODO: Shouldn't we do a QR here? Need to reduce the number of column
    # vectors in that case.
    for k in range(b):
        X = op(X)

    # Since the vectors are all row vectors, we need to consider the right
    # singular vectors.
    _, S, V = np.linalg.svd(X, full_matrices=False)
    V = V[:K]

    V = np.reshape(V, (K,) + sig_sz)

    return V


def estimate_psd_periodogram(x, d):
    sig_sz = x.shape[-d:]

    xf = np.fft.fftn(x, axes=range(-d, 0))

    x_per = 1 / np.prod(sig_sz) * np.abs(xf) ** 2

    return x_per


def estimate_psd_rand_tapers(x, mask, W=1/8, p=5, b=3, gen_fun=None):
    d = mask.ndim

    sig_sz = mask.shape

    h = calc_rand_tapers(mask, W=W, p=p, b=b, gen_fun=gen_fun)

    taper_len = h.shape[0]

    x_rt = np.zeros_like(x)

    for h_m in h:
        x_tapered = x * h_m

        x_rt += (1 / taper_len * np.prod(sig_sz)
                 * estimate_psd_periodogram(x_tapered, d))

    return x_rt


def _ensure_W(W, d):
    W = np.array(W)

    if W.ndim == 0:
        W = W[np.newaxis]

    if W.shape[0] == 1:
        W = W.repeat(d)
    elif W.shape[0] != d:
        raise TypeError('Bandwidth W must have 1 or d elements.')

    if any(W >= 0.5):
        raise ValueError('Bandwidth W must be strictly smaller than 0.5.')

    return W

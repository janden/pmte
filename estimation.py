import numpy as np


def calc_rand_tapers(mask, W=1/8, p=5, b=3, gen_fun=None):
    if gen_fun is None:
        rng = np.random.default_rng()
        gen_fun = rng.standard_normal

    d = mask.ndim

    W = np.array(W)
    if W.ndim == 0:
        W = W[np.newaxis]
    if W.shape[0] == 1:
        W = W.repeat(d)
    elif W != d:
        raise TypeError('Bandwidth W must have 1 or d elements.')
    if any(W >= 0.5):
        raise ValueError('Bandwidth W must be strictly smaller than 0.5.')

    sig_sz = mask.shape
    sig_len = np.prod(sig_sz)

    rngs = [np.fft.fftfreq(sz) for sz in sig_sz]
    grids = np.meshgrid(*rngs, indexing='ij')

    freq_mask = np.full(sig_sz, True)

    for ell in range(d):
        freq_mask &= np.abs(grids[ell]) < W[ell]

    K = int(np.ceil(np.sum(freq_mask) / sig_len * np.sum(mask)))

    # TODO: Does this actually apply T? Looks more like it applies a Dirichlet
    # kernel since it just masks in the DFT domain.
    def _apply(x):
        x = np.reshape(x, x.shape[:1] + sig_sz)

        x = x * mask

        xf = np.fft.fftn(x, axes=range(-d, 0))

        xf = xf * freq_mask

        x = np.fft.ifftn(xf, axes=range(-d, 0))
        x = np.real(x)

        x = x * mask

        x = np.reshape(x, x.shape[:1] + (sig_len,))

        return x

    X = gen_fun((K + p, sig_len))

    # TODO: Shouldn't we do a QR here? Need to reduce the number of column
    # vectors in that case.
    for k in range(b):
        X = _apply(X)

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

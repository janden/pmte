import numpy as np
from scipy.fft import fftn, ifftn
from scipy.linalg import qr
from scipy.signal.windows import dpss

from pmte import _internal


def concentration_op(mask, W=1 / 4, use_fftw=True):
    if use_fftw:
        pyfftw, use_fftw = _internal.try_import_pyfftw()

    d = mask.ndim

    W = _ensure_W(W, d)

    block_size = 64
    n_threads = 8

    sig_sz = mask.shape
    sig_len = np.prod(sig_sz)

    two_sig_sz = tuple(2 * sz for sz in sig_sz)

    rngs = [sz * np.fft.fftfreq(sz) for sz in two_sig_sz]
    grids = np.meshgrid(*rngs, indexing='ij')

    sinc_kernel = np.ones(two_sig_sz)

    for ell in range(d):
        sinc_kernel *= W[ell] * np.sinc(W[ell] * grids[ell])

    freq_mask = fftn(sinc_kernel, axes=range(-d, 0), workers=-1)

    fft_sig_sz = two_sig_sz

    if use_fftw:
        in_array = pyfftw.empty_aligned((block_size,) + fft_sig_sz,
                                        dtype='float64')
        out_array = pyfftw.empty_aligned((block_size,) + fft_sig_sz[:-1]
                                         + (fft_sig_sz[-1] // 2 + 1,),
                                         dtype='complex128')

        plan_forward = pyfftw.FFTW(in_array, out_array, axes=range(-d, 0),
                                   direction='FFTW_FORWARD',
                                   flags=('FFTW_MEASURE',),
                                   threads=n_threads)

        plan_backward = pyfftw.FFTW(out_array, in_array, axes=range(-d, 0),
                                    direction='FFTW_BACKWARD',
                                    flags=('FFTW_MEASURE',),
                                    threads=n_threads)

        freq_mask = freq_mask[..., :out_array.shape[-1]]

    def _apply(x):
        x = np.reshape(x, x.shape[:1] + sig_sz)

        x = x * mask

        ixgrid = np.ix_(range(x.shape[0]), *(range(sz) for sz in sig_sz))

        if use_fftw:
            in_array[:] = 0
            in_array[ixgrid] = x
            plan_forward()

            # We have to use np.multiply here instead of out_array *= because
            # otherwise Python assigns out_array to local scope instead of
            # using the outer variable defined in concentration_op.
            np.multiply(out_array, freq_mask, out=out_array)

            plan_backward()
            x = in_array
        else:
            y = np.zeros(x.shape[:1] + two_sig_sz)
            y[ixgrid] = x
            x = y

            xf = fftn(x, axes=range(-d, 0), workers=-1)
            xf = xf * freq_mask
            x = ifftn(xf, axes=range(-d, 0), workers=-1)
            x = np.real(x)

        x = x[ixgrid]

        x = x * mask

        x = np.reshape(x, x.shape[:1] + (sig_len,))

        return x

    def _blocked_apply(x):
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


def tensor_tapers(sig_shape, W=1 / 4):
    if not isinstance(sig_shape, tuple):
        sig_shape = (sig_shape, )

    d = len(sig_shape)

    W = _ensure_W(W, d)

    h = np.ones((1,) * (2 * d), dtype=np.float64)

    K = np.round(np.array(sig_shape) * W).astype('int')

    for ell in range(d):
        if K[ell] > 0:
            h_ell = dpss(sig_shape[ell], sig_shape[ell] * W[ell] / 2,
                         Kmax=K[ell], norm=2)
        else:
            # If K is too small, just use constant taper (so we get the
            # periodogram).
            h_ell = 1 / np.sqrt(sig_shape[ell]) * np.ones((1, sig_shape[ell]))

        # Move first and second axes into ellth and (d + ell)th,
        # respectively.
        new_axes = (tuple(range(0, ell))
                    + tuple(range(ell + 1, d))
                    + tuple(range(d, d + ell))
                    + tuple(range(d + ell + 1, 2 * d)))
        h_ell = np.expand_dims(h_ell, new_axes)

        h = h * h_ell

    taper_shape = h.shape[:d]
    taper_len = np.prod(taper_shape)

    h = np.reshape(h, (taper_len, ) + sig_shape)

    return h


def corner_tapers(mask, W=1 / 4):
    d = mask.ndim

    if not d == 2:
        raise RuntimeError('Only implemented for 2D signals.')

    N1, N2, N3, N4 = _fit_corner_mask(mask)

    tapers1 = tensor_tapers((N1,) * 2, W=W)
    tapers2 = tensor_tapers((N2,) * 2, W=W)
    tapers3 = tensor_tapers((N3,) * 2, W=W)
    tapers4 = tensor_tapers((N4,) * 2, W=W)

    taper_len = (tapers1.shape[0]
                 + tapers2.shape[0]
                 + tapers3.shape[0]
                 + tapers4.shape[0])

    tapers = np.zeros((taper_len,) + mask.shape, dtype=tapers1.dtype)

    idx = 0

    tapers[idx:idx + tapers1.shape[0], :N1, :N1] = tapers1
    idx += tapers1.shape[0]

    tapers[idx:idx + tapers2.shape[0], -N2:, :N2] = tapers2
    idx += tapers2.shape[0]

    tapers[idx:idx + tapers3.shape[0], -N3:, -N3:] = tapers3
    idx += tapers3.shape[0]

    tapers[idx:idx + tapers4.shape[0], :N4, -N4:] = tapers4

    return tapers


def _orthogonalize(X):
    Q, _ = qr(X.T, mode='economic', check_finite=False, overwrite_a=True)
    return Q.T


def proxy_tapers(mask, W=1 / 4, n_iter=8, K=None, rng=None, use_fftw=True):
    if rng is None:
        rng = np.random.default_rng()

    qr_period = 16

    d = mask.ndim

    W = _ensure_W(W, d)

    sig_sz = mask.shape
    sig_len = np.prod(sig_sz)

    if K is None:
        K = int(np.ceil(np.prod(W) * np.sum(mask)))

    op = concentration_op(mask, W=W, use_fftw=use_fftw)

    X = rng.standard_normal((K, sig_len))

    for k in range(n_iter):
        if k % qr_period == 0:
            X = _orthogonalize(X)

        X = op(X)

    V = _orthogonalize(X)

    V = np.reshape(V, (K,) + sig_sz)

    return V


def taper_intensity(tapers, grid_sz=None, shifted=False):
    d = tapers.ndim - 1

    if grid_sz is None:
        grid_sz = tapers.shape[-d:]

    tapers_f = fftn(tapers, grid_sz, axes=range(-d, 0), workers=-1)

    if shifted:
        tapers_f = np.fft.fftshift(tapers_f, axes=range(-d, 0))

    inten = np.sum(np.abs(tapers_f) ** 2, axis=0)
    inten /= tapers.shape[0]

    return inten


def _ensure_W(W, d):
    W = np.array(W)

    if W.ndim == 0:
        W = W[np.newaxis]

    if W.shape[0] == 1:
        W = W.repeat(d)
    elif W.shape[0] != d:
        raise TypeError('Bandwidth W must have 1 or d elements.')

    if any(W >= 1):
        raise ValueError('Bandwidth W must be strictly smaller than 1.')

    return W


def _fit_corner_mask(mask):
    def last_true(x):
        """Return index of last true element"""
        return np.argmax(~x)

    # Side lengths of squares fitting in the top left, bottom left, bottom
    # right, and top right corners of the mask.

    N1 = last_true(np.diag(mask))
    N2 = last_true(np.diag(np.flip(mask, 0)))
    N3 = last_true(np.diag(np.flip(mask, (0, 1))))
    N4 = last_true(np.diag(np.flip(mask, 1)))

    return (N1, N2, N3, N4)

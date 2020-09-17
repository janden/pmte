import numpy as np

from scipy.fft import fftn, ifftn

import tapers
import _internal


def estimate_psd_periodogram(x, d):
    sig_sz = x.shape[-d:]

    xf = fftn(x, axes=range(-d, 0), workers=-1)

    x_per = 1 / np.prod(sig_sz) * (xf.real ** 2 + xf.imag ** 2)

    return x_per


def estimate_psd_rand_tapers(x, mask, W=1 / 4, n_iter=8,
        use_fftw=True, rng=None):
    h = tapers.proxy_tapers(mask, W, n_iter=n_iter, rng=rng,
            use_fftw=use_fftw)

    x_rt = estimate_psd_tapers(x, h, use_fftw=use_fftw)

    return x_rt


def estimate_psd_multitaper(x, d, W=1 / 4, use_fftw=True):
    shape = x.shape[-d:]

    h = tapers.tensor_tapers(shape, W)

    x_mt = estimate_psd_tapers(x, h, use_fftw=use_fftw)

    return x_mt


def estimate_psd_tapers(x, tapers, use_fftw=True):
    if use_fftw:
        pyfftw, use_fftw = _internal.try_import_pyfftw()

    d = tapers.ndim - 1

    if d > 2 and use_fftw:
        raise RuntimeError('FFTW is not supported for d > 2')

    sig_sz = tapers.shape[-d:]
    taper_len = tapers.shape[0]

    precision = np.real(x.ravel()[0]).dtype.itemsize

    real_dtype = {4: 'float32', 8: 'float64'}[precision]
    complex_dtype = {4: 'complex64', 8: 'complex128'}[precision]

    is_real = (np.real(x.ravel()[0]).dtype == x.dtype)

    if use_fftw:
        n_threads = 8

        if is_real:
            in_array = pyfftw.empty_aligned(x.shape,
                                            dtype=real_dtype)
            out_array = pyfftw.empty_aligned(x.shape[:-1]
                                            + (x.shape[-1] // 2 + 1,),
                                            dtype=complex_dtype)
        else:
            in_array = pyfftw.empty_aligned(x.shape,
                                            dtype=complex_dtype)
            out_array = pyfftw.empty_aligned(x.shape,
                                             dtype=complex_dtype)

        x_mt = np.zeros(out_array.shape, dtype=real_dtype)

        plan = pyfftw.FFTW(in_array, out_array, axes=range(-d, 0),
                           direction='FFTW_FORWARD',
                           flags=('FFTW_MEASURE',),
                           threads=n_threads)

        for taper in tapers:
            np.multiply(x, taper, out=in_array)

            plan()

            np.add(x_mt, out_array.real ** 2 + out_array.imag ** 2, out=x_mt)

        if is_real:
            # NOTE: The below only works for d = 1 and d = 2. For higher
            # dimensions, we need to handle the Nyquist frequency separately
            # (a Fourier, or Hermitian, flip).

            # Otherwise the indexing below will destroy the original x_mt.
            x_mt_flip = x_mt[..., -2:0:-1].copy()

            # Don't want to flip the first (i.e., Nyquist) frequency.
            ixgrid = np.ix_(*(range(sz) for sz in x_mt_flip.shape[:-d]), *(range(1,
                sz) for sz in x_mt_flip.shape[-d:-1]), range(x_mt_flip.shape[-1]))

            x_mt_flip[ixgrid] = np.flip(x_mt_flip[ixgrid], axis=range(-d, -1))

            x_mt = np.concatenate((x_mt, x_mt_flip), axis=-1)
    else:
        x_mt = np.zeros_like(x, dtype=real_dtype)
        x_tapered = np.empty_like(x)

        for taper in tapers:
            np.multiply(x, taper, out=x_tapered)

            x_tapered_f = fftn(x_tapered, axes=range(-d, 0), workers=-1)

            np.add(x_mt, x_tapered_f.real ** 2 + x_tapered_f.imag ** 2, out=x_mt)

    x_mt /= taper_len

    return x_mt

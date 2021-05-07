import numpy as np


def grid(sz, normalized=True, shifted=False):
    sz = np.array(sz)

    if sz.ndim == 0:
        sz = sz[np.newaxis]

    rngs = [np.ceil(np.arange(-N / 2, N / 2)) for N in sz]

    if normalized:
        rngs = [rng / N for rng, N in zip(rngs, sz)]

    if not shifted:
        rngs = [np.fft.ifftshift(rng) for rng in rngs]

    grid = np.meshgrid(*rngs, indexing='ij')

    return grid

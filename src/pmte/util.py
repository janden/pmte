import os

import numpy as np


def log_slope(x, y):
    x = np.log(x)
    y = np.log(y)

    x = x - np.mean(x)
    y = y - np.mean(y)

    beta = np.dot(x, y) / np.dot(x, x)

    return beta


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


def target_win(Nf, W, shifted=False):
    fX, fY = grid((Nf, Nf), shifted=shifted)

    rho = np.zeros((Nf, Nf))

    half_W = W / 2

    rho[(np.abs(fX) < half_W) & (np.abs(fY) < half_W)] = 1 / W ** 2
    rho[(np.abs(fX) == half_W) & (np.abs(fY) < half_W)] = 1 / W ** 2 / 2
    rho[(np.abs(fX) < half_W) & (np.abs(fY) == half_W)] = 1 / W ** 2 / 2
    rho[(np.abs(fX) == half_W) & (np.abs(fY) == half_W)] = 1 / W ** 2 / 4

    return rho


def disk_mask(N, R):
    x1, x2 = grid((N, N), normalized=False, shifted=True)

    r = np.hypot(x1, x2)

    mask = (r < R)

    return mask


def square_mask(N, R):
    x1, x2 = grid((N, N), normalized=False, shifted=True)

    mask = (-R <= x1) & (x1 < R) & (-R <= x2) & (x2 < R)

    return mask

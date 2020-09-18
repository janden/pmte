import sys
import os

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dirname, '..', 'src'))

import json

import numpy as np

from pmte import estimation, simulation, tapers, util


def main():
    N = 128
    W = 1 / 8
    R2 = 1 / 3

    mask2 = ~util.disk_mask(N, N * R2)

    rng = np.random.default_rng(0)

    density_fun = lambda xi1, xi2: \
        np.exp(-80 * (xi1 - 0.20) ** 2 - 40 * (xi2 - 0.25) ** 2) \
        + np.exp(-40 * (xi1 + 0.25) ** 2 - 80 * (xi2 + 0.25) ** 2) \
        + 1.44 * np.exp(-80 * (xi1 - 0.10) ** 2 - 40 * (xi2 + 0.10) ** 2)

    xi1, xi2 = util.grid((N, N), shifted=True)
    density = density_fun(xi1, xi2)

    def calc_error(ref, est):
        error = np.sqrt(np.mean(np.abs(est - ref) ** 2))
        error /= np.sqrt(np.mean(np.abs(ref) ** 2))

        return error

    fname = 'data/density.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, density)

    signal = simulation.generate_field((N, N), 1, psd_fun=density_fun,
            rng=rng, real=False)
    signal = signal[0]

    proxy_tapers = tapers.proxy_tapers(mask2, W, rng=rng)

    inten = tapers.taper_intensity(proxy_tapers, shifted=True)

    fname = 'data/tap1.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, proxy_tapers[0, :, :])

    fname = 'data/tap2.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, proxy_tapers[1, :, :])

    fname = 'data/tap17.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, proxy_tapers[16, :, :])

    fname = 'data/inten.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, inten)

    multiestim = estimation.multitaper(signal, proxy_tapers)
    multiestim = np.fft.fftshift(multiestim, axes=(-2, -1))

    fname = 'data/mt.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, multiestim)

    error = calc_error(density, multiestim)

    results = {'error': float(error)}

    with open('data/tap.json', 'w') as f:
        json.dump(results, f)
        f.write('\n')

if __name__ == '__main__':
    main()

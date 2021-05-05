import _preamble

import numpy as np

from pmte import estimation, simulation, tapers, util

import helpers


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

    helpers.save_image('density', density)

    signal = simulation.generate_field((N, N), 1, psd_fun=density_fun,
            rng=rng, real=False)
    signal = signal[0]

    proxy_tapers = tapers.proxy_tapers(mask2, W, rng=rng)

    inten = tapers.taper_intensity(proxy_tapers, shifted=True)

    helpers.save_image('tap1', proxy_tapers[0])
    helpers.save_image('tap2', proxy_tapers[1])
    helpers.save_image('tap17', proxy_tapers[16])

    helpers.save_image('inten', inten)

    multiestim = estimation.multitaper(signal, proxy_tapers)
    multiestim = np.fft.fftshift(multiestim, axes=(-2, -1))

    helpers.save_image('mt', multiestim)

    error = calc_error(density, multiestim)

    results = {'error': float(error)}

    helpers.save_dictionary('tap', results)

if __name__ == '__main__':
    main()

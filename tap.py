import numpy as np

import estimation
import simulation
import compat
import util


def main():
    N = 128
    R1 = 1 / 16
    R2 = 1 / 3

    g1d = np.arange(-N // 2, N // 2) / N

    x1, x2 = np.meshgrid(g1d, g1d)
    xi1, xi2 = x1, x2

    mask2 = np.hypot(xi1, xi2) >= R2

    gen_fun = compat.oct_randn()

    W = R1

    tapers = estimation.calc_rand_tapers(mask2, W, p=0, b=8,
                                         gen_fun=gen_fun, use_sinc=True,
                                         use_fftw=True)

    inten = estimation.taper_intensity(tapers)

    fname = 'data/tap1.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tapers[0, :, :].T)

    fname = 'data/tap2.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tapers[1, :, :].T)

    fname = 'data/tap17.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tapers[16, :, :].T)

    fname = 'data/inten.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, inten.T)

    density_fun = lambda xi1, xi2: \
        np.exp(-80 * (xi1 - 0.20) ** 2 - 40 * (xi2 - 0.25) ** 2) \
        + np.exp(-40 * (xi1 + 0.25) ** 2 - 80 * (xi2 + 0.25) ** 2) \
        + 1.44 * np.exp(-80 * (xi1 - 0.10) ** 2 - 40 * (xi2 + 0.10) ** 2)

    density = density_fun(xi1, xi2)

    fname = 'data/density.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, density.T)

    signal = simulation.generate_field((N, N), 1, psd_fun=density_fun,
            gen_fun=gen_fun, real=False)
    signal = signal[0]

    multiestim = estimation.estimate_psd_tapers(signal, tapers)
    multiestim = np.fft.fftshift(multiestim, axes=(-2, -1))

    fname = 'data/mt.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, multiestim.T)

    error = np.sqrt(np.sum(np.abs(density.ravel() - multiestim.ravel()) ** 2)
            / np.sum(np.abs(density.ravel()) ** 2))

    print(error)


if __name__ == '__main__':
    main()

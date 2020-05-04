import numpy as np

import estimation
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

    sqrtdensity = lambda xi1, xi2: \
        np.exp(-40 * (xi1 - 0.20) ** 2 - 20 * (xi2 - 0.25) ** 2) \
        + np.exp(-20 * (xi1 + 0.25) ** 2 - 40 * (xi2 + 0.25) ** 2) \
        + 1.2 * np.exp(-40 * (xi1 - 0.10) ** 2 - 40 * (xi2 + 0.10) ** 2)

    # TODO: Make this work with simulation.generate_field.
    sqrtdensity = sqrtdensity(xi2, xi1)
    density = sqrtdensity ** 2

    signal = gen_fun((N, N))
    signal = util.centered_ifftn(sqrtdensity * util.centered_fftn(signal, 2), 2)

    multiestim = estimation.estimate_psd_tapers(signal, tapers)
    multiestim = np.fft.fftshift(multiestim, axes=(-2, -1))

    fname = 'data/mt.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, multiestim.T)

    error = (np.sum(np.abs(density.ravel() - multiestim.ravel()) ** 2)
            / np.sum(np.abs(density.ravel()) ** 2))

    print(error)


if __name__ == '__main__':
    main()

import numpy as np
import json

import estimation
import util
import simulation


def main():
    N = 128
    W = 1 / 8
    R = N / 3

    recmask2 = util.square_mask(N, R)

    rng = np.random.default_rng(0)
    gen_fun = rng.standard_normal

    density_fun = lambda xi1, xi2: \
        np.exp(-80 * (xi1 - 0.20) ** 2 - 40 * (xi2 - 0.25) ** 2) \
        + np.exp(-40 * (xi1 + 0.25) ** 2 - 80 * (xi2 + 0.25) ** 2) \
        + 1.44 * np.exp(-80 * (xi1 - 0.10) ** 2 - 40 * (xi2 + 0.10) ** 2)

    xi1, xi2 = util.grid((N, N), normalized=True)
    density = density_fun(xi1, xi2)

    signal = simulation.generate_field((N, N), 1, psd_fun=density_fun,
            gen_fun=gen_fun, real=False)
    signal = signal[0]

    K = int(np.ceil(np.sqrt(np.sum(recmask2)) * W)) ** 2

    rectapers = estimation.calc_rand_tapers(recmask2, W, b=8, K=K,
                                            gen_fun=gen_fun,
                                            use_fftw=True)

    recinten = estimation.taper_intensity(rectapers)

    fname = 'data/rectap1.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, rectapers[0, :, :])

    fname = 'data/rectap2.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, rectapers[1, :, :])

    fname = 'data/rectap17.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, rectapers[16, :, :])

    fname = 'data/recinten.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, recinten)

    recmultiestim = estimation.estimate_psd_tapers(signal, rectapers)
    recmultiestim = np.fft.fftshift(recmultiestim, axes=(-2, -1))

    fname = 'data/recmt.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, recmultiestim)

    recerror = np.sqrt(np.sum(np.abs(density.ravel() - recmultiestim.ravel()) ** 2)
                       / np.sum(np.abs(density.ravel()) ** 2))

    N_tensor = int(np.floor(2 * R))

    tapers = estimation.calc_tensor_tapers((N_tensor, N_tensor), W=W)

    tentapers = np.zeros((tapers.shape[0], N, N))

    mid = int(np.ceil((N + 1) / 2) - 1)
    ext1 = int(np.ceil((N_tensor - 1) / 2))
    ext2 = int(np.floor((N_tensor - 1) / 2) + 1)

    tentapers[:, mid - ext1:mid + ext2, mid - ext1:mid + ext2] = tapers

    teninten = estimation.taper_intensity(tentapers)

    fname = 'data/tentap1.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tentapers[0, :, :])

    fname = 'data/tentap2.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tentapers[1, :, :])

    fname = 'data/tentap17.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tentapers[16, :, :])

    fname = 'data/teninten.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, teninten)

    tenmultiestim = estimation.estimate_psd_tapers(signal, tentapers)
    tenmultiestim = np.fft.fftshift(tenmultiestim, axes=(-2, -1))

    fname = 'data/tenmt.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tenmultiestim)

    tenerror = np.sqrt(np.sum(np.abs(density.ravel() - tenmultiestim.ravel()) ** 2)
                       / np.sum(np.abs(density.ravel()) ** 2))

    deviation = np.sqrt(np.sum(np.abs(tenmultiestim.ravel() - recmultiestim.ravel()) ** 2)
                        / np.sum(np.abs(tenmultiestim.ravel()) ** 2))

    rectapers_conv = estimation.calc_rand_tapers(recmask2, W, b=72, K=K,
                                                 gen_fun=gen_fun,
                                                 use_fftw=True)

    recmultiestim_conv = estimation.estimate_psd_tapers(signal, rectapers_conv)
    recmultiestim_conv = np.fft.fftshift(recmultiestim_conv, axes=(-2, -1))

    deviation_conv = \
            np.sqrt(np.sum(np.abs(tenmultiestim - recmultiestim_conv) ** 2)
                    / np.sum(np.abs(tenmultiestim) ** 2))

    results = {'recerror': float(recerror),
               'tenerror': float(tenerror),
               'deviation': float(deviation),
               'deviation_conv': float(deviation_conv)}

    with open('data/rectap.json', 'w') as f:
        json.dump(results, f)
        f.write('\n')


if __name__ == '__main__':
    main()

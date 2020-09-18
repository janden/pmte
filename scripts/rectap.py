import _preamble

import json

import numpy as np

from pmte import estimation, simulation, tapers, util


def main():
    N = 128
    W = 1 / 8
    R = N / 3

    recmask2 = util.square_mask(N, R)

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

    signal = simulation.generate_field((N, N), 1, psd_fun=density_fun,
            rng=rng, real=False)
    signal = signal[0]

    K = int(np.ceil(np.sqrt(np.sum(recmask2)) * W)) ** 2

    rectapers = tapers.proxy_tapers(recmask2, W, K=K, rng=rng)

    recinten = tapers.taper_intensity(rectapers, shifted=True)

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

    recmultiestim = estimation.multitaper(signal, rectapers)
    recmultiestim = np.fft.fftshift(recmultiestim, axes=(-2, -1))

    fname = 'data/recmt.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, recmultiestim)

    recerror = calc_error(density, recmultiestim)

    N_tensor = int(np.floor(2 * R))

    smalltapers = tapers.tensor_tapers((N_tensor, N_tensor), W)

    pad = (int(np.ceil((N - N_tensor) / 2)),
           int(np.floor((N - N_tensor) / 2)))

    tentapers = np.pad(smalltapers, ((0, 0),) + (pad,) * 2)

    teninten = tapers.taper_intensity(tentapers, shifted=True)

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

    tenmultiestim = estimation.multitaper(signal, tentapers)
    tenmultiestim = np.fft.fftshift(tenmultiestim, axes=(-2, -1))

    fname = 'data/tenmt.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tenmultiestim)

    tenerror = calc_error(density, tenmultiestim)

    deviation = calc_error(tenmultiestim, recmultiestim)

    rectapers_conv = tapers.proxy_tapers(recmask2, W, n_iter=72, K=K, rng=rng)

    recmultiestim_conv = estimation.multitaper(signal, rectapers_conv)
    recmultiestim_conv = np.fft.fftshift(recmultiestim_conv, axes=(-2, -1))

    deviation_conv = calc_error(tenmultiestim, recmultiestim_conv)

    results = {'recerror': float(recerror),
               'tenerror': float(tenerror),
               'deviation': float(deviation),
               'deviation_conv': float(deviation_conv)}

    with open('data/rectap.json', 'w') as f:
        json.dump(results, f)
        f.write('\n')


if __name__ == '__main__':
    main()

import numpy as np
import json

from scipy.signal.windows import dpss

import estimation
import compat
import util
import simulation


def slepian_tapers(n, W):
    V, E = dpss(n, n * W, Kmax=n, norm=2, return_ratios=True)
    V = V.T

    return V, E


def main():
    N = 128
    R1 = 1 / 16
    R2 = 1 / 3

    g1d = np.arange(-N // 2, N // 2) / N

    x1, x2 = np.meshgrid(g1d, g1d, indexing='ij')
    xi1, xi2 = x1, x2

    recmask2 = (-R2 <= xi1) & (xi1 < R2) & (-R2 <= xi2) & (xi2 < R2)

    rng = np.random.default_rng(0)
    gen_fun = rng.standard_normal

    W = R1

    K = int(np.ceil(np.sqrt(np.sum(recmask2)) * 2 * W)) ** 2

    rectapers = estimation.calc_rand_tapers(recmask2, W, p=0, b=72, K=K,
                                            gen_fun=gen_fun, use_sinc=True,
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

    density_fun = lambda xi1, xi2: \
        np.exp(-80 * (xi1 - 0.20) ** 2 - 40 * (xi2 - 0.25) ** 2) \
        + np.exp(-40 * (xi1 + 0.25) ** 2 - 80 * (xi2 + 0.25) ** 2) \
        + 1.44 * np.exp(-80 * (xi1 - 0.10) ** 2 - 40 * (xi2 + 0.10) ** 2)

    density = density_fun(xi1, xi2)

    signal = simulation.generate_field((N, N), 1, psd_fun=density_fun,
            gen_fun=gen_fun, real=False)
    signal = signal[0]

    recmultiestim = estimation.estimate_psd_tapers(signal, rectapers)
    recmultiestim = np.fft.fftshift(recmultiestim, axes=(-2, -1))

    fname = 'data/recmt.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, recmultiestim)

    recerror = np.sqrt(np.sum(np.abs(density.ravel() - recmultiestim.ravel()) ** 2)
                       / np.sum(np.abs(density.ravel()) ** 2))

    N_tensor = int(np.floor(2 * R2 * N))
    tapers, _ = slepian_tapers(N_tensor, R1)

    K = int(np.round(2 * N_tensor * R1))

    tapers = tapers[:, :K]

    tapers = (tapers[:, np.newaxis, :, np.newaxis]
              * tapers[np.newaxis, :, np.newaxis, :])

    tapers = np.reshape(tapers, (N_tensor, N_tensor, -1))

    tentapers = np.zeros((N, N, tapers.shape[2]))

    mid = int(np.ceil((N + 1) / 2) - 1)
    ext1 = int(np.ceil((N_tensor - 1) / 2))
    ext2 = int(np.floor((N_tensor - 1) / 2) + 1)

    tentapers[mid - ext1:mid + ext2, mid - ext1:mid + ext2, :] = tapers.real

    teninten = estimation.taper_intensity(tentapers.T).T

    fname = 'data/tentap1.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tentapers[:, :, 0].T)

    fname = 'data/tentap2.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tentapers[:, :, 1].T)

    fname = 'data/tentap17.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tentapers[:, :, 16].T)

    fname = 'data/teninten.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, teninten)

    tenmultiestim = estimation.estimate_psd_tapers(signal, tentapers.T)
    tenmultiestim = np.fft.fftshift(tenmultiestim, axes=(-2, -1))

    fname = 'data/tenmt.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tenmultiestim)

    tenerror = np.sqrt(np.sum(np.abs(density.ravel() - tenmultiestim.ravel()) ** 2)
                       / np.sum(np.abs(density.ravel()) ** 2))

    deviation = np.sqrt(np.sum(np.abs(tenmultiestim.ravel() - recmultiestim.ravel()) ** 2)
                        / np.sum(np.abs(tenmultiestim.ravel()) ** 2))

    results = {'recerror': float(recerror),
               'tenerror': float(tenerror),
               'deviation': float(deviation)}

    with open('data/rectap.json', 'w') as f:
        json.dump(results, f)
        f.write('\n')


if __name__ == '__main__':
    main()

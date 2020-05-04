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

    recmask2 = (-R2 <= xi1) & (xi1 < R2) & (-R2 <= xi2) & (xi2 < R2)

    gen_fun = compat.oct_randn()

    W = R1

    K = int(np.ceil(np.sqrt(np.sum(recmask2)) * 2 * W)) ** 2

    tapers = estimation.calc_rand_tapers(recmask2, W, p=0, b=8, K=K,
                                         gen_fun=gen_fun, use_sinc=True,
                                         use_fftw=True)

    fname = 'data/rectap1.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tapers[0, :, :].T)

    fname = 'data/rectap2.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tapers[1, :, :].T)

    fname = 'data/rectap17.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tapers[16, :, :].T)


if __name__ == '__main__':
    main()

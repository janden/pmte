import numpy as np

import util


def main():
    N = 128
    R1 = 1 / 16
    R2 = 1 / 3

    g1d = np.arange(-N // 2, N // 2) / N

    x1, x2 = np.meshgrid(g1d, g1d)
    xi1, xi2 = x1, x2

    mask1 = (np.abs(x1) < R1) & (np.abs(x2) < R1)

    fname = 'data/mask1.bin'

    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, mask1)

    mask2 = np.hypot(xi1, xi2) >= R2

    fname = 'data/mask2.bin'

    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, mask2)

    recmask2 = (-R2 <= xi1) & (xi1 < R2) & (-R2 <= xi2) & (xi2 < R2)

    fname = 'data/recmask2.bin'

    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, recmask2)

if __name__ == '__main__':
    main()

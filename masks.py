import numpy as np

import util


def main():
    N = 128
    W = 1 / 8
    R = N / 3

    mask1 = util.target_win(N, W, shifted=True)

    fname = 'data/mask1.bin'

    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, mask1)

    mask2 = ~util.disk_mask(N, R)

    fname = 'data/mask2.bin'

    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, mask2)

    recmask2 = util.square_mask(N, R)

    fname = 'data/recmask2.bin'

    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, recmask2)

if __name__ == '__main__':
    main()

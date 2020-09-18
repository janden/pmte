import _preamble

import numpy as np

from pmte import util

import datahelpers


def main():
    N = 128
    W = 1 / 8
    R = N / 3

    mask1 = util.target_win(N, W, shifted=True)

    datahelpers.save_image('mask1', mask1)

    mask2 = ~util.disk_mask(N, R)

    datahelpers.save_image('mask2', mask2)

    recmask2 = util.square_mask(N, R)

    datahelpers.save_image('recmask2', recmask2)

if __name__ == '__main__':
    main()

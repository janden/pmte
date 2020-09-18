import _preamble

import numpy as np

from pmte.tapers import concentration_op

import datahelpers


def main():
    N = 32
    W = 7 / 32

    mask = np.full((N,), True)
    op = concentration_op(mask, W)

    A = op(np.eye(N))

    D = np.linalg.eigvalsh(A)

    D = D[::-1]

    idx = np.arange(1, len(D) + 1)

    datahelpers.save_table('spectrum', idx, D)


if __name__ == '__main__':
    main()

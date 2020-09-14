import numpy as np

from estimation import concentration_op
import util


def main():
    N = 32
    W = 7 / 32

    mask = np.full((N,), True)
    op = concentration_op(mask, W=W)

    A = op(np.eye(N))

    D = np.linalg.eigvalsh(A)

    D = D[::-1]

    fname = 'data/spectrum.csv'

    util.ensure_dir_exists(fname)

    with open(fname, 'w') as f:
        for k, lam in enumerate(D):
            f.write('%d %.15g\n' % (k + 1, lam))


if __name__ == '__main__':
    main()

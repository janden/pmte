import numpy as np

import util


def main():
    N = 32
    W = 0.1

    g1d = np.fft.fftfreq(N)
    g1dp = np.fft.fftfreq(2 * N)

    A = 1 / np.pi * np.sinc(np.pi * W * N
                            * (g1d[np.newaxis, :] - g1d[:, np.newaxis]))

    D, _ = np.linalg.eig(A)

    D = np.real(D)

    D = np.sort(D)[::-1]

    fname = 'data/spectrum.csv'

    util.ensure_dir_exists(fname)

    with open(fname, 'w') as f:
        for k, lam in enumerate(D):
            f.write('%d %.15g\n' % (k + 1, lam))


if __name__ == '__main__':
    main()

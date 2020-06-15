import numpy as np
import json

import util
from estimation import calc_rand_tapers


def main():
    N = 256

    W = 1/8

    X, Y = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-N / 2, N / 2))
    R = np.sqrt(X ** 2 + Y ** 2)

    rs = 2 ** np.linspace(-4, -1, 3 * 4 + 1)

    fX = np.fft.ifftshift(X) / N
    fY = np.fft.ifftshift(Y) / N

    err1 = []

    rng = np.random.default_rng(0)
    gen_fun = rng.standard_normal

    err1 = np.empty_like(rs)

    for k, r in enumerate(rs):
        mask = R < (r * N)
        h = calc_rand_tapers(mask, W, gen_fun=gen_fun, use_fftw=True,
                use_sinc=True, p=0, b=8)
        rho = 1 / h.shape[0] * np.sum(np.abs(np.fft.fft2(h)) ** 2, axis=0)

        rho0 = np.zeros_like(rho)
        rho0[(np.abs(fX) < W) & (np.abs(fY) < W)] = 1 / (2 * W) ** 2
        rho0[(np.abs(fX) == W) & (np.abs(fY) < W)] = 1 / (2 * W) ** 2 / 2
        rho0[(np.abs(fX) < W) & (np.abs(fY) == W)] = 1 / (2 * W) ** 2 / 2
        rho0[(np.abs(fX) == W) & (np.abs(fY) == W)] = 1 / (2 * W) ** 2 / 4

        err1[k] = np.linalg.norm(rho.ravel() - rho0.ravel(), 1) / rho.size

    fname = 'data/rho1_single.csv'

    util.ensure_dir_exists(fname)

    with open(fname, 'w') as f:
        for r, err in zip(rs, err1):
            f.write('%.15g %.15g\n' % (r * N, err))

    beta = util.log_slope(rs * N, err1)

    results = {'beta': float(beta)}

    with open('data/rho1_single.json', 'w') as f:
        json.dump(results, f)
        f.write('\n')


if __name__ == '__main__':
    main()

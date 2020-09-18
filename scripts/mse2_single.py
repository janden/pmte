import _preamble

import numpy as np
from scipy.integrate import fixed_quad
from scipy.special import jv

from pmte import estimation, simulation, tapers, util

import datahelpers


def main():
    N = 128
    n = 200

    def psd_fun(x, y):
        quadrature = lambda func: fixed_quad(func, 0, 100, n=200)[0]

        kernel = lambda u, xi: (jv(0, 8 * xi[..., np.newaxis] * u)
                                * (jv(1, u) / u) ** 3 * u)

        r = np.hypot(x, y)

        c = quadrature(lambda u: kernel(u, np.zeros(1)))

        density = 1 / c * quadrature(lambda u: kernel(u, r))

        return density

    rng = np.random.default_rng(0)

    X = simulation.generate_field((N, N), n, psd_fun=psd_fun, rng=rng)

    rs = 2 ** np.linspace(-4, -1, 3 * 4 + 1)

    err2 = []
    caro2 = []

    xi1, xi2 = util.grid((N, N))
    psd_true = psd_fun(xi1, xi2)

    for r in rs:
        mask = util.disk_mask(N, r * N)
        nmask = np.sum(mask)

        W = nmask ** (-1 / 6)

        h = tapers.proxy_tapers(mask, W, rng=rng)
        X_rt = estimation.multitaper(X, h)

        err2.append(np.max(np.mean(np.abs(psd_true - X_rt) ** 2, axis=0)))

        caro2.append(nmask)

    err2 = np.array(err2)
    caro2 = np.array(caro2)

    datahelpers.save_table('mse2_single', rs * N, err2)

    beta = util.log_slope(rs * N, err2 / np.log(caro2) ** 2)

    results = {'beta': float(beta)}

    datahelpers.save_json('mse2_single', results)


if __name__ == '__main__':
    main()

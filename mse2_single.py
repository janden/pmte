import numpy as np
import json

from scipy.special import jv
from scipy.integrate import fixed_quad

import simulation
import estimation
import util


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
    gen_fun = rng.standard_normal

    X = simulation.generate_field((N, N), n, psd_fun=psd_fun,
                                  gen_fun=gen_fun)

    rs = 2 ** np.linspace(-4, -1, 3 * 4 + 1)

    err2 = np.empty_like(rs)
    caro2 = np.empty_like(rs)

    xi1, xi2 = util.grid((N, N), normalized=True)
    psd_true = psd_fun(xi1, xi2)

    for k, r in enumerate(rs):
        mask = util.disk_mask(N, r * N)
        nmask = np.sum(mask)

        W = nmask ** (-1 / 6)

        X_rt = estimation.estimate_psd_rand_tapers(X, mask, W=W,
                                                   gen_fun=gen_fun,
                                                   use_fftw=True)

        err2[k] = np.max(np.mean(np.abs(psd_true - X_rt) ** 2, axis=0))

        caro2[k] = nmask

    fname = 'data/mse2_single.csv'

    util.ensure_dir_exists(fname)

    with open(fname, 'w') as f:
        for r, err in zip(rs, err2):
            f.write('%.15g %.15g\n' % (r * N, err))

    beta = util.log_slope(rs * N, err2 / np.log(caro2) ** 2)

    results = {'beta': float(beta)}

    with open('data/mse2_single.json', 'w') as f:
        json.dump(results, f)
        f.write('\n')


if __name__ == '__main__':
    main()

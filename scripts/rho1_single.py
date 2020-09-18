import _preamble

import numpy as np

from pmte import tapers, util

import datahelpers


def main():
    N = 128

    Nf = 2 * N

    W = 1 / 4

    rs = 2 ** np.linspace(-4, -1, 3 * 4 + 1)

    rng = np.random.default_rng(0)

    err1 = []

    for r in rs:
        mask = util.disk_mask(N, r * N)
        h = tapers.proxy_tapers(mask, W, rng=rng)

        rho = tapers.taper_intensity(h, grid_sz=(Nf, Nf))

        rho0 = util.target_win(Nf, W)

        err1.append(np.mean(np.abs(rho - rho0)))

    err1 = np.array(err1)

    datahelpers.save_table('rho1_single', rs * N, err1)

    beta = util.log_slope(rs * N, err1)

    results = {'beta': float(beta)}

    datahelpers.save_json('rho1_single', results)


if __name__ == '__main__':
    main()

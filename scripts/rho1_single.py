import _preamble

import json

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

    fname = 'data/rho1_single.csv'

    datahelpers.ensure_dir_exists(fname)

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

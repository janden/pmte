import numpy as np
import json

from scipy.special import jv

import oct2py

import simulation
import estimation
import util


def main():
    N = 128
    n = 200

    X, Y = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-N / 2, N / 2))
    R = np.sqrt(X ** 2 + Y ** 2)

    fX = np.fft.ifftshift(X) / N
    fY = np.fft.ifftshift(Y) / N

    # TODO: These weights do not seem to correspond exactly to those of lgwt.
    omegan, omegaw = np.polynomial.legendre.leggauss(200)
    omegan = (omegan + 1) / 2 * 100
    omegaw = omegaw / 2 * 100

    def _psd_fun(x, y):
        r = np.hypot(x, y)

        c = np.sum(omegaw * omegan * (jv(1, omegan) / omegan) ** 3)

        return 1 / c * np.sum(jv(0, 8 * r[:, np.newaxis] * omegan) * omegaw * omegan
                              * (jv(1, omegan) / omegan) ** 3, axis=-1)

    def _vectorized_psd_fun(*coords):
        sz = coords[0].shape

        coords = [coord.ravel() for coord in coords]

        val = _psd_fun(*coords)

        val = np.reshape(val, sz)

        return val

    rng = np.random.default_rng(0)
    gen_fun = rng.standard_normal

    X = simulation.generate_field((N, N), n, psd_fun=_vectorized_psd_fun,
                                  gen_fun=gen_fun)

    rs = 2 ** np.linspace(-4, -1, 3 * 4 + 1)

    err2 = np.empty_like(rs)
    caro2 = np.empty_like(rs)

    psd_true = _vectorized_psd_fun(fX, fY)

    for k, r in enumerate(rs):
        mask = R < (r * N)
        nmask = np.sum(mask)

        W = 1 / 2 * nmask ** (-1 / 6)

        X_rt = estimation.estimate_psd_rand_tapers(X, mask, W=W,
                                                   b=8,
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

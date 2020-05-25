import numpy as np

import util
import compat
import estimation


def main():
    n = 1024

    W = 1 / 128

    mask_rs = np.arange(36, 212 + 1, 8) / 360

    do_print = True

    x, proj, groups = util.load_exp_images(n)

    N = x.shape[-1]

    g1d = np.arange(-N // 2, N // 2) / N
    x1, x2 = np.meshgrid(g1d, g1d)
    r = np.hypot(x1, x2)

    gen_fun = compat.oct_randn()

    psd_true = estimation.estimate_psd_multitaper(x - proj, 2, W=W)

    def mse(psd_est):
        return np.mean(np.abs(psd_est - psd_true) ** 2)

    if do_print:
        print('%-20s%15s' % ('Estimator', 'MSE'))

    mses_rt = []
    mses_mper = []
    mses_cmt = []

    for mask_r in mask_rs:
        mask = (r >= mask_r)

        x_rt = estimation.estimate_psd_rand_tapers(x, mask, W=W, gen_fun=gen_fun, p=0)

        mse_rt = mse(x_rt)

        mses_rt.append(mse_rt)

        if do_print:
            print('%-20s%15e' % ('Randomtaper', mse_rt))

        xm = x * mask

        x_mper = estimation.estimate_psd_periodogram(xm, 2)
        x_mper *= N ** 2 / np.sum(mask)

        mse_mper = mse(x_mper)

        mses_mper.append(mses_mper)

        if do_print:
            print('%-20s%15e' % ('Mask periodogram', mse_mper))

        tapers = estimation.calc_corner_tapers(mask, W=W)
        x_cmt = estimation.estimate_psd_tapers(x, tapers)

        mse_cmt = mse(x_cmt)

        mses_cmt.append(mses_cmt)

        if do_print:
            print('%-20s%15e' % ('Corner multitaper', mse_cmt))

    with open('data/cryo_exp.csv', 'w') as f:
        for k in range(len(mask_rs)):
            f.write('%d %g %g %g\n' % (round(N * mask_rs[k]), mses_mper[k],
                                       mses_cmt[k], mses_rt[k]))


if __name__ == '__main__':
    main()

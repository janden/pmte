import numpy as np
import json

import util
import estimation
import tapers


def main():
    n = 120

    W = 1 / 16

    mask_rs = np.arange(36, 144 + 1, 12) / 360

    do_print = True

    x, proj = util.load_exp_images(n)

    util.write_gplt_binary_matrix('data/cryo_exp_sig_noise1.bin', x[0])
    util.write_gplt_binary_matrix('data/cryo_exp_sig_noise2.bin', x[1])

    N = x.shape[-1]

    rng = np.random.default_rng(0)

    h = tapers.tensor_tapers((N, N), W)
    psd_true = estimation.estimate_psd_tapers(x - proj, h)

    def mse(psd_est):
        return np.mean(np.abs(psd_est - psd_true) ** 2)

    if do_print:
        print('%-20s%15s' % ('Estimator', 'MSE'))

    mses_rt = []
    mses_mper = []
    mses_cmt = []

    for mask_r in mask_rs:
        mask = ~util.disk_mask(N, mask_r * N)

        h = tapers.proxy_tapers(mask, W, rng=rng)
        x_rt = estimation.estimate_psd_tapers(x, h)

        mse_rt = mse(x_rt)

        mses_rt.append(mse_rt)

        if do_print:
            print('%-20s%15e' % ('Randomtaper', mse_rt))

        xm = x * mask

        x_mper = estimation.estimate_psd_periodogram(xm, 2)
        x_mper *= N ** 2 / np.sum(mask)

        mse_mper = mse(x_mper)

        mses_mper.append(mse_mper)

        if do_print:
            print('%-20s%15e' % ('Mask periodogram', mse_mper))

        corner_tapers = tapers.corner_tapers(mask, W)
        x_cmt = estimation.estimate_psd_tapers(x, corner_tapers)

        mse_cmt = mse(x_cmt)

        mses_cmt.append(mse_cmt)

        if do_print:
            print('%-20s%15e' % ('Corner multitaper', mse_cmt))

    with open('data/cryo_exp.csv', 'w') as f:
        for k in range(len(mask_rs)):
            f.write('%d %g %g %g\n' % (round(N * mask_rs[k]), mses_mper[k],
                                       mses_cmt[k], mses_rt[k]))

    min_mse_mper = np.min(mses_mper)
    min_mse_cmt = np.min(mses_cmt)
    min_mse_rt = np.min(mses_rt)

    argmin_mse_rt = 360 * mask_rs[np.argmin(mses_rt)]

    mse_factor_mper = min_mse_mper / min_mse_rt
    mse_factor_cmt = min_mse_cmt / min_mse_rt

    results = {'min_mse_mper': float(min_mse_mper),
               'min_mse_cmt': float(min_mse_cmt),
               'min_mse_rt': float(min_mse_rt),
               'argmin_mse_rt': float(argmin_mse_rt),
               'mse_factor_mper': float(mse_factor_mper),
               'mse_factor_cmt': float(mse_factor_cmt)}

    with open('data/cryo_exp.json', 'w') as f:
        json.dump(results, f)
        f.write('\n')

if __name__ == '__main__':
    main()

import numpy as np
import json

import util
import compat
import estimation


def main():
    n = 120

    W = 1 / 32

    mask_rs = np.arange(36, 172 + 1, 8) / 360

    do_print = True

    x, proj = util.load_new_exp_images(n)

    util.write_gplt_binary_matrix('data/cryo_exp_sig_noise1.bin', x[0])
    util.write_gplt_binary_matrix('data/cryo_exp_sig_noise2.bin', x[1])

    N = x.shape[-1]

    g1d = np.arange(-N // 2, N // 2) / N
    x1, x2 = np.meshgrid(g1d, g1d)
    r = np.hypot(x1, x2)

    rng = np.random.default_rng(0)
    gen_fun = rng.standard_normal

    psd_true = estimation.estimate_psd_multitaper(x - proj, 2, W=W,
            use_fftw=True)

    def mse(psd_est):
        return np.mean(np.abs(psd_est - psd_true) ** 2)

    if do_print:
        print('%-20s%15s' % ('Estimator', 'MSE'))

    mses_rt = []
    mses_mper = []
    mses_cmt = []

    for mask_r in mask_rs:
        mask = (r >= mask_r)

        x_rt = estimation.estimate_psd_rand_tapers(x, mask, W=W,
                gen_fun=gen_fun, p=0, b=8, use_sinc=True, use_fftw=True)

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

        tapers = estimation.calc_corner_tapers(mask, W=W)
        x_cmt = estimation.estimate_psd_tapers(x, tapers, use_fftw=True)

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

    mse_factor_mper = min_mse_mper / min_mse_rt
    mse_factor_cmt = min_mse_cmt / min_mse_rt

    results = {'min_mse_rt': float(min_mse_rt),
               'mse_factor_mper': float(mse_factor_mper),
               'mse_factor_cmt': float(mse_factor_cmt)}

    with open('data/cryo_exp.json', 'w') as f:
        json.dump(results, f)
        f.write('\n')

if __name__ == '__main__':
    main()

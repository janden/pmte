import _preamble

import numpy as np

from pmte import estimation, tapers, util

import helpers


def main():
    n = 120

    W = 1 / 16

    mask_rs = np.arange(36, 144 + 1, 12) / 360

    do_print = True

    x, proj = helpers.load_exp_images(n)

    helpers.save_image('cryo_exp_sig_noise1', x[0])
    helpers.save_image('cryo_exp_sig_noise2', x[1])

    N = x.shape[-1]

    rng = np.random.default_rng(0)

    h = tapers.tensor_tapers((N, N), W)
    psd_true = estimation.multitaper(x - proj, h)

    def calc_mse(psd_est):
        return np.mean(np.abs(psd_est - psd_true) ** 2)

    def est_mper(x, mask):
        x_mper = estimation.periodogram(x * mask, 2)
        x_mper *= N ** 2 / np.sum(mask)

        return x_mper

    def est_pmt(x, mask):
        h_pmt = tapers.proxy_tapers(mask, W, rng=rng)
        x_pmt = estimation.multitaper(x, h_pmt)

        return x_pmt

    def est_cmt(x, mask):
        h_cmt = tapers.corner_tapers(mask, W)
        x_cmt = estimation.multitaper(x, h_cmt)

        return x_cmt

    methods = {'mper': est_mper, 'cmt': est_cmt, 'pmt': est_pmt}

    mses = {name: [] for name in methods}

    if do_print:
        print('%-20s%15s' % ('Estimator', 'MSE'))

    for mask_r in mask_rs:
        mask = ~util.disk_mask(N, mask_r * N)

        for name, method in methods.items():
            x_est = method(x, mask)

            mse = calc_mse(x_est)

            if do_print:
                print('%-20s%15e' % (name, mse))

            mses[name].append(mse)

    helpers.save_table('cryo_exp', np.round(N * mask_rs), mses['mper'],
                           mses['cmt'], mses['pmt'])

    min_mse_mper = np.min(mses['mper'])
    min_mse_cmt = np.min(mses['cmt'])
    min_mse_pmt = np.min(mses['pmt'])

    argmin_mse_pmt = 360 * mask_rs[np.argmin(mses['pmt'])]

    mse_factor_mper = min_mse_mper / min_mse_pmt
    mse_factor_cmt = min_mse_cmt / min_mse_pmt

    results = {'min_mse_mper': float(min_mse_mper),
               'min_mse_cmt': float(min_mse_cmt),
               'min_mse_rt': float(min_mse_pmt),
               'argmin_mse_rt': float(argmin_mse_pmt),
               'mse_factor_mper': float(mse_factor_mper),
               'mse_factor_cmt': float(mse_factor_cmt)}

    helpers.save_dictionary('cryo_exp', results)

if __name__ == '__main__':
    main()

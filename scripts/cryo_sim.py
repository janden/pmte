import _preamble

import numpy as np

from pmte import estimation, simulation, tapers, util

import datahelpers


def main():
    N = 128
    n = 1000

    width = 0.125

    W = 1 / 8

    mask_rs = np.arange(36, 72 + 1, 4) / 128

    do_print = True

    psd_fun = lambda x, y: np.exp(-np.hypot(x, y) ** 2 / (2 * width ** 2))

    rng = np.random.default_rng(0)

    x = simulation.generate_field((N, N), n,
            psd_fun=psd_fun, rng=rng)

    sig = datahelpers.load_sim_images(n)

    x = (x + 10 * sig).astype(sig.dtype)

    xi1, xi2 = util.grid((N, N))
    psd_true = psd_fun(xi1, xi2)

    datahelpers.save_image('cryo_sim_sig1', sig[0])
    datahelpers.save_image('cryo_sim_sig2', sig[1])
    datahelpers.save_image('cryo_sim_sig_noise1', x[0])
    datahelpers.save_image('cryo_sim_sig_noise2', x[1])
    datahelpers.save_image('cryo_sim_psd',
                           np.fft.ifftshift(psd_true, axes=(-2, -1)))

    mask_r = 60 / 128
    mask = ~util.disk_mask(N, mask_r * N)
    datahelpers.save_image('cryo_sim_mask', mask)

    corner_tapers = tapers.corner_tapers(mask, W)
    corner_mask = np.any(np.abs(corner_tapers) > 0, axis=0)
    datahelpers.save_image('cryo_sim_mask_grid', corner_mask)

    def calc_mse(psd_est):
        return np.mean(np.abs(psd_est - psd_true) ** 2)

    def calc_bias(psd_est):
        return np.mean(np.abs(np.mean(psd_est, axis=0) - psd_true) ** 2)

    def calc_variance(psd_est):
        return np.mean(np.abs(psd_est - np.mean(psd_est, axis=0)) ** 2)

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
    variances = {name: [] for name in methods}
    biases = {name: [] for name in methods}

    for mask_r in mask_rs:
        mask = ~util.disk_mask(N, mask_r * N)

        for name, method in methods.items():
            x_est = method(x, mask)

            mse = calc_mse(x_est)
            bias = calc_bias(x_est)
            variance = calc_variance(x_est)

            if do_print:
                print('%-20s%15e%15e%15e' % (name, mse, bias, variance))

            mses[name].append(mse)
            biases[name].append(bias)
            variances[name].append(variance)

    datahelpers.save_table('cryo_sim_biases', np.round(N * mask_rs),
                           biases['mper'], biases['cmt'], biases['pmt'])

    datahelpers.save_table('cryo_sim_variances', np.round(N * mask_rs),
                           variances['mper'], variances['cmt'],
                           variances['pmt'])

    datahelpers.save_table('cryo_sim_mses', np.round(N * mask_rs),
                           mses['mper'], mses['cmt'],
                           mses['pmt'])

    min_mse_mper = np.min(mses['mper'])
    min_mse_cmt = np.min(mses['cmt'])
    min_mse_pmt = np.min(mses['pmt'])

    mse_factor_mper = min_mse_mper / min_mse_pmt
    mse_factor_cmt = min_mse_cmt / min_mse_pmt

    min_variance_mper = np.min(variances['mper'])
    min_variance_cmt = np.min(variances['cmt'])
    min_variance_pmt = np.min(variances['pmt'])

    variance_factor_mper = min_variance_mper / min_variance_pmt
    variance_factor_cmt = min_variance_cmt / min_variance_pmt

    results = {'min_mse_rt': float(min_mse_pmt),
               'mse_factor_mper': float(mse_factor_mper),
               'mse_factor_cmt': float(mse_factor_cmt),
               'variance_factor_mper': float(variance_factor_mper),
               'variance_factor_cmt': float(variance_factor_cmt)}

    datahelpers.save_json('cryo_sim', results)


if __name__ == '__main__':
    main()

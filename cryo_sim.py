import numpy as np
import json

import estimation
import simulation
import util


def main():
    N = 128
    n = 500

    width = 0.125

    W = 1 / 16

    mask_rs = np.arange(28, 76 + 1, 4) / 128

    do_print = True

    g1d = np.arange(-N // 2, N // 2) / N
    x1, x2 = np.meshgrid(g1d, g1d)
    r = np.hypot(x1, x2)

    psd_fun = lambda r: np.exp(-r ** 2 / (2 * width ** 2))

    rng = np.random.default_rng(0)
    gen_fun = rng.standard_normal

    x = simulation.generate_field((N, N), n,
            psd_fun=lambda x, y: psd_fun(np.hypot(x, y)),
            gen_fun=gen_fun)

    sig = util.load_new_sim_images(n)

    x = (x + 10 * sig).astype(sig.dtype)

    psd_true = psd_fun(np.fft.fftshift(r, axes=(-2, -1)))

    util.write_gplt_binary_matrix('data/cryo_sim_sig1.bin', sig[0])
    util.write_gplt_binary_matrix('data/cryo_sim_sig2.bin', sig[1])
    util.write_gplt_binary_matrix('data/cryo_sim_sig_noise1.bin', x[0])
    util.write_gplt_binary_matrix('data/cryo_sim_sig_noise2.bin', x[1])
    util.write_gplt_binary_matrix('data/cryo_sim_psd.bin',
                                  np.fft.ifftshift(psd_true, axes=(-2, -1)))

    mask_r = 60 / 128
    mask = (r >= mask_r)
    util.write_gplt_binary_matrix('data/cryo_sim_mask.bin', mask)

    corner_tapers = estimation.calc_corner_tapers(mask, W=W)
    corner_mask = np.any(np.abs(corner_tapers) > 0, axis=0)
    util.write_gplt_binary_matrix('data/cryo_sim_mask_grid.bin', mask)

    def mse(psd_est):
        return np.mean(np.abs(psd_est - psd_true) ** 2)

    def bias(psd_est):
        return np.mean(np.abs(np.mean(psd_est, axis=0) - psd_true) ** 2)

    def variance(psd_est):
        return np.mean(np.abs(psd_est - np.mean(psd_est, axis=0)) ** 2)

    mses_mper = []
    variances_mper = []
    biases_mper = []

    mses_cmt = []
    variances_cmt = []
    biases_cmt = []

    mses_rt = []
    variances_rt = []
    biases_rt = []

    for mask_r in mask_rs:
        mask = (r >= mask_r)

        xm = x * mask

        x_mper = estimation.estimate_psd_periodogram(xm, 2)
        x_mper *= N ** 2 / np.sum(mask)

        mse_mper = mse(x_mper)
        bias_mper = bias(x_mper)
        variance_mper = variance(x_mper)

        mses_mper.append(mse_mper)
        variances_mper.append(variance_mper)
        biases_mper.append(bias_mper)

        if do_print:
            print('%-20s%15e%15e%15e' % ('Masked periodogram', mse_mper, bias_mper, variance_mper))

        x_rt = estimation.estimate_psd_rand_tapers(x, mask, W=W, p=0, b=8,
                use_sinc=True, use_fftw=True, gen_fun=gen_fun)

        mse_rt = mse(x_rt)
        bias_rt = bias(x_rt)
        variance_rt = variance(x_rt)

        mses_rt.append(mse_rt)
        variances_rt.append(variance_rt)
        biases_rt.append(bias_rt)

        if do_print:
            print('%-20s%15e%15e%15e' % ('Randomtaper', mse_rt, bias_rt, variance_rt))

        tapers = estimation.calc_corner_tapers(mask, W=W)
        x_cmt = estimation.estimate_psd_tapers(x, tapers, use_fftw=True)

        mse_cmt = mse(x_cmt)
        bias_cmt = bias(x_cmt)
        variance_cmt = variance(x_cmt)

        mses_cmt.append(mse_cmt)
        variances_cmt.append(variance_cmt)
        biases_cmt.append(bias_cmt)

        if do_print:
            print('%-20s%15e%15e%15e' % ('Corner multitaper', mse_cmt, bias_cmt, variance_cmt))

    with open('data/cryo_sim.csv', 'w') as f:
        for k in range(len(mask_rs)):
            f.write('%d %g %g %g\n' % (round(N * mask_rs[k]), biases_mper[k],
                                       biases_cmt[k], biases_rt[k]))
        f.write('\n\n')

        for k in range(len(mask_rs)):
            f.write('%d %g %g %g\n' % (round(N * mask_rs[k]), variances_mper[k],
                                       variances_cmt[k], variances_rt[k]))
        f.write('\n\n')

        for k in range(len(mask_rs)):
            f.write('%d %g %g %g\n' % (round(N * mask_rs[k]), mses_mper[k],
                                       mses_cmt[k], mses_rt[k]))

    min_mse_mper = np.min(mses_mper)
    min_mse_cmt = np.min(mses_cmt)
    min_mse_rt = np.min(mses_rt)

    mse_factor_mper = min_mse_mper / min_mse_rt
    mse_factor_cmt = min_mse_cmt / min_mse_rt

    min_variance_mper = np.min(variances_mper)
    min_variance_cmt = np.min(variances_cmt)
    min_variance_rt = np.min(variances_rt)

    variance_factor_mper = min_variance_mper / min_variance_rt
    variance_factor_cmt = min_variance_cmt / min_variance_rt

    results = {'min_mse_rt': float(min_mse_rt),
               'mse_factor_mper': float(mse_factor_mper),
               'mse_factor_cmt': float(mse_factor_cmt),
               'variance_factor_mper': float(variance_factor_mper),
               'variance_factor_cmt': float(variance_factor_cmt)}

    with open('data/cryo_sim.json', 'w') as f:
        json.dump(results, f)
        f.write('\n')


if __name__ == '__main__':
    main()

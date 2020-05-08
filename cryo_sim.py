import numpy as np

import estimation
import simulation
import compat
import util


def main():
    N = 128
    n = 1000

    width = 0.125

    W = 1 / 16

    mask_r = 56 / 128

    g1d = np.arange(-N // 2, N // 2) / N
    x1, x2 = np.meshgrid(g1d, g1d)
    r = np.hypot(x1, x2)

    gen_fun = compat.oct_randn()

    psd_fun = lambda r: np.exp(-r ** 2 / (2 * width ** 2))

    x = simulation.generate_field((N, N), n,
            psd_fun=lambda x, y: psd_fun(np.hypot(x, y)),
            gen_fun=gen_fun)

    sig = util.load_images(n)

    x = (x + 1 / 4 * sig).astype(sig.dtype)

    mask = (r >= mask_r)

    psd_true = psd_fun(np.fft.fftshift(r, axes=(-2, -1)))

    def mse(psd_est):
        return np.mean(np.abs(psd_est - psd_true) ** 2)

    def bias(psd_est):
        return np.mean(np.abs(np.mean(psd_est, axis=0) - psd_true) ** 2)

    def variance(psd_est):
        return np.mean(np.abs(psd_est - np.mean(psd_est, axis=0)) ** 2)

    xm = x * mask

    x_mper = estimation.estimate_psd_periodogram(xm, 2)
    x_mper *= N ** 2 / np.sum(mask)

    mse_mper = mse(x_mper)
    bias_mper = bias(x_mper)
    variance_mper = variance(x_mper)

    print('%-20s%15e%15e%15e' % ('Masked periodogram', mse_mper, bias_mper, variance_mper))

    x_rt = estimation.estimate_psd_rand_tapers(x, mask, W=W, gen_fun=gen_fun)

    mse_rt = mse(x_rt)
    bias_rt = bias(x_rt)
    variance_rt = variance(x_rt)

    print('%-20s%15e%15e%15e' % ('Randomtaper', mse_rt, bias_rt, variance_rt))

    tapers = estimation.calc_corner_tapers(mask, W=W)
    x_cmt = estimation.estimate_psd_tapers(x, tapers)

    mse_cmt = mse(x_cmt)
    bias_cmt = bias(x_cmt)
    variance_cmt = variance(x_cmt)

    print('%-20s%15e%15e%15e' % ('Corner multitaper', mse_cmt, bias_cmt, variance_cmt))


if __name__ == '__main__':
    main()

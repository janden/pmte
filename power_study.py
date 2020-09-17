import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import subspace_angles

from tapers import concentration_op
from scipy.signal.windows import dpss


def gaussian_prob(n):
    kappa = 256

    rng = np.random.default_rng(0)

    lam = np.exp(-np.arange(n) ** 2 / (2 * (kappa) ** 2))

    D = np.diag(lam)

    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    A = Q @ D @ Q.T

    op = lambda x: A @ x

    return op, Q, lam


def init_subspace(n, r):
    rng = np.random.default_rng(0)

    V = rng.standard_normal((n, r))
    V, _ = np.linalg.qr(V, 'reduced')

    return V


def slepian_tapers(n, W):
    V, E = dpss(n, int(n * W), Kmax=n, norm=2, return_ratios=True)
    V = V.T

    return V, E

def tensor_tapers(n, W):
    V0, E0 = slepian_tapers(n, W)

    V = V0[:, np.newaxis, :, np.newaxis] * V0[np.newaxis, :, np.newaxis, :]
    E = E0[:, np.newaxis] * E0[ np.newaxis, :]

    V = np.reshape(V, (n ** 2,) * 2)
    E = np.reshape(E, (n ** 2,))

    idx = np.argsort(E)[::-1]

    V = V[:, idx]
    E = E[idx]

    return V, E


def concentration_prob_1d(n, use_fft=False):
    W = 1 / 4

    V, E = slepian_tapers(n, W)

    if use_fft:
        op_orig = concentration_op(np.full(n, True), W)
        op = lambda x: op_orig(x.T).T
    else:
        op = lambda x: V @ (E[:, np.newaxis] * (V.T @ x))

    return op, V, E


def concentration_prob_2d(n, use_fft=False):
    W = 1 / 4

    V, E = tensor_tapers(n, W)

    if use_fft:
        op_orig = concentration_op(np.full((n,) * 2, True), W)
        op = lambda x: op_orig(x.T).T
    else:
        op = lambda x: V @ (E[:, np.newaxis] * (V.T @ x))

    return op, V, E


def main():
    n = 256
    max_iter = 100 + 1
    r = 64 - 1

    problem = 'concentration2d'

    sampling_step = 10

    if problem == 'gaussian':
        op, Q0, lam0 = gaussian_prob(n)
    elif problem == 'concentration1d':
        op, Q0, lam0 = concentration_prob_1d(n, use_fft=True)
    elif problem == 'concentration2d':
        op, Q0, lam0 = concentration_prob_2d(int(np.sqrt(n)), use_fft=True)
    else:
        raise RuntimeError('Unknown problem')

    spectral_gap = lam0[r] / lam0[r-1]

    print('%-30s[%.8g, %.8g, %.8g, %.8g, %.8g]' % ('Local spectrum:', *lam0[r-3:r+2]))

    print('%-30s%.15g' % ('Spectral gap:', spectral_gap))

    V0 = init_subspace(n, r)
    theta0 = np.max(subspace_angles(V0, Q0[:, :r]))

    err = np.empty((max_iter - 1) // sampling_step + 1)

    V = V0

    for k in range(max_iter):
        V = op(V)
        V, _ = np.linalg.qr(V, 'reduced')

        if k % sampling_step == 0:
            sample = k // sampling_step
            err[sample] = np.sin(np.max(subspace_angles(V, Q0[:, :r])))

    if all(err > 1e-10):
        last_good = -1
    else:
        last_good = np.nonzero(err < 1e-10)[0][0] - 1
    asymp_rate = (err[last_good] / err[last_good - 1]) ** (1 / sampling_step)

    print('%-30s%.15g' % ('Asymptotic rate:', asymp_rate))

    bound = (np.tan(theta0)
             * spectral_gap ** np.arange(0, max_iter, sampling_step))

    plt.figure()
    plt.semilogy(np.arange(0, max_iter, sampling_step), err, 'o-', label='error')
    plt.semilogy(np.arange(0, max_iter, sampling_step), bound, 's-', label='bound')
    plt.ylim((np.min(err), max(np.max(err), np.max(bound))))
    plt.xlim((0, max_iter - 1))
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()

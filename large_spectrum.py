import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigs
from time import time
from scipy.linalg import subspace_angles

from estimation import concentration_op, calc_rand_tapers
import util


def main():
    N = 128

    W = 1 / 8

    n_eigs = 256

    qr_period = 32
    save_period = 128

    error_type = 'trace_norm'

    X, Y = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-N / 2, N / 2))
    R = np.hypot(X, Y)

    r = 43

    mask = R > r

    compare_ref = False

    op = concentration_op(mask, W=W, use_fftw=True)

    if compare_ref:
        h = calc_rand_tapers(mask, W=W, p=0, b=32)

        # Need this to be a standard 2D matrix.
        h = np.reshape(h, (h.shape[0], -1))

    K = int(np.ceil((2 * W) ** 2 * np.sum(mask)))

    print('K = %d' % K)

    rng = np.random.default_rng()
    gen_fun = rng.standard_normal
    X = gen_fun((K, N ** 2))

    errs = []

    for k in range(0, 5120 + 1):
        X_prev = X
        t0 = time()
        X = op(X)
        t1 = time()
        if k % qr_period == 0:
            X = np.linalg.qr(X.T, 'reduced')[0].T
        t2 = time()

        print('[k=%04d] timing = %.2g, %.2g' % (k, t1-t0, t2-t1))

        if k > 0 and k % qr_period == 0:
            angles = subspace_angles(X.T, X_prev.T)
            theta = np.max(angles)

            if error_type == 'operator_norm':
                error = np.sin(theta)
            elif error_type == 'trace_norm':
                error = np.mean(np.sin(angles))

            print('[k=%04d] error = %.15g' % (k, error))

            if len(errs) > 0:
                print('[k=%04d] rate = %.15g' % (k, error / errs[-1]))

            errs.append(error)

        if k % save_period == 0:
            np.save('subspace_%04d.npy' % k, X)

    err = np.array(errs)

    if len(err) >= 2:
        if all(err > 1e-10):
            last_good = -1
        else:
            last_good = np.nonzero(err < 1e-10)[0][0] - 1
        asymp_rate = (err[last_good] / err[last_good - 1])

        print('%-30s%.15g' % ('Asymptotic rate:', asymp_rate))

        plt.figure()
        plt.semilogy(qr_period * np.arange(1, len(err) + 1), err)
        plt.title('N = 128, R = 8, W = 1/8, K = %d' % K)
        plt.ylabel('Error')
        plt.xlabel('Iteration (t)')
        plt.savefig('conv_K=%d.png' % K)
        plt.show()

    if compare_ref:
        corr = np.min(np.linalg.svd(X @ h.T)[1])

        print('corr = %.15g' % corr)

        dist = util.subspace_dist(X, h)

        print('dist = %.15g' % dist)

    h = X

    # The eigs routine expects input vectors to be column vectors, so add the
    # necessary transposes to convert to row vectors (which op is expecting).
    op_transp = lambda x: op(x.T).T

    # Project on the orthogonal complement of taper subspace.
    proj_compl = lambda x: x - h.T @ (h @ x)

    op_subspace = lambda x: h @ op_transp(h.T @ x)
    op_compl = lambda x: proj_compl(op_transp(proj_compl(x)))

    linop_subspace = LinearOperator(2 * (h.shape[0],), matvec=op_subspace,
                                    dtype=np.float64)
    linop_compl = LinearOperator(2 * (h.shape[1],), matvec=op_compl,
                                 dtype=np.float64)

    lams_subspace, eig_subspace = eigs(linop_subspace, n_eigs, which='SM')
    lams_compl, eig_compl = eigs(linop_compl, n_eigs, which='LM')

    eig_subspace = np.real(eig_subspace)
    eig_compl = np.real(eig_compl)

    M = np.real(eig_subspace.T @ op_subspace(eig_subspace))
    nondiag_subspace = np.linalg.norm(M - np.diag(np.diag(M)))

    np.save('M_subspace.npy', M)

    M = np.real(eig_compl.T @ op_compl(eig_compl))
    nondiag_compl = np.linalg.norm(M - np.diag(np.diag(M)))

    np.save('M_compl.npy', M)

    print('%-30s%.15g' % ('Non-diagonal norm (subspace):', nondiag_subspace))
    print('%-30s%.15g' % ('Non-diagonal norm (compl):', nondiag_compl))

    lams_subspace = np.real(lams_subspace)
    lams_subspace = np.sort(lams_subspace)[::-1]

    lams_compl = np.real(lams_compl)
    lams_compl = np.sort(lams_compl)[::-1]

    spectral_gap_est = lams_compl[0] / lams_subspace[-1]

    print('%-30s%.15g' % ('Estimated gap:', spectral_gap_est))

    fname = 'data/large_spectrum.csv'

    util.ensure_dir_exists(fname)

    with open(fname, 'w') as f:
        for k, lam in enumerate(lams_subspace):
            f.write('%d %.15g\n' % (k + 1, lam))

        f.write('\n')

        for k, lam in enumerate(lams_compl):
            f.write('%d %.15g\n' % (k + n_eigs + 1, lam))


def load_spectrum():
    fname = 'data/large_spectrum.csv'

    lams = []

    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                continue

            lams.append(float(line.split()[1]))

    return lams


def load_errs():
    fname = 'data/errors.csv'

    errs = []

    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()

            errs.append(float(line))

    return errs


def calc_error():
    fname = 'subspace_%04d.npy'
    max_iter = 5120
    step = 128

    error_type = 'trace_norm'

    lams = load_spectrum()
    X0 = np.load(fname % max_iter)

    K = X0.shape[0]

    plt.figure(figsize=(18, 6))
    lams_range = np.arange(K - len(lams) // 2 + 1, K + len(lams) // 2 + 1)
    plt.bar(lams_range, lams, color=[0, 0, 0.7])
    plt.bar(lams_range[len(lams) // 2], lams[len(lams) // 2], color=[0.8, 0, 0])
    plt.title('')
    plt.ylabel('λ[k]')
    plt.xlabel('k')
    plt.xlim((lams_range[0], lams_range[-1]))
    plt.ylim((0, 1.2))
    plt.title('Eigenvalues of T around k = %d' % K)
    plt.savefig('spectrum_full.png')
    plt.xlim((lams_range[len(lams) // 2 - 32], lams_range[len(lams) // 2 + 32]))
    plt.savefig('spectrum_zoom.png')

    if True:
        err = np.empty(max_iter // step + 1)

        for k in range(0, max_iter // step + 1):
            X = np.load(fname % (k * step))

            angles = subspace_angles(X.T, X0.T)

            if error_type == 'operator_norm':
                error = np.sin(np.max(angles))
            elif error_type == 'trace_norm':
                error = np.mean(np.sin(angles))

            err[k] = error

        fname = 'data/errors.csv'

        with open(fname, 'w') as f:
            for err_k in err:
                f.write('%.15g\n' % err_k)
    else:
        err = load_errs()

    # Not the same seed, but should give similar theta.
    rng = np.random.default_rng()
    gen_fun = rng.standard_normal
    X_init = gen_fun(X0.shape)

    theta0 = np.max(subspace_angles(X_init.T, X0.T))
    spectral_gap_est = lams[len(lams) // 2] / lams[len(lams) // 2 - 1]

    it = np.arange(0, max_iter + 1, step)

    bound0 = np.tan(theta0) * spectral_gap_est ** it

    plt.figure()
    plt.semilogy(it, err, 'o-', label='error')
    plt.semilogy(it, bound0, 's-', label='bound')
    plt.title('Subspace distance as a function of iteration')
    plt.ylabel('Error')
    plt.xlabel('t')
    plt.legend()
    plt.savefig('conv.png')

if __name__ == '__main__':
    main()
    calc_error()

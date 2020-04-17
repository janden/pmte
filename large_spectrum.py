import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from estimation import concentration_op, calc_rand_tapers
import util


def main():
    N = 128

    W = 1 / 8

    n_eigs = 1

    X, Y = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-N / 2, N / 2))
    R = np.hypot(X, Y)

    r = 43

    #mask = R > r
    mask = R < 8

    compare_ref = False

    op = concentration_op(mask, W=W, use_sinc=True)

    h = calc_rand_tapers(mask, W=W, p=0, b=32, use_sinc=True)

    # Need this to be a standard 2D matrix.
    h = np.reshape(h, (h.shape[0], -1))

    K = h.shape[0] + 1

    print('K = %d' % K)

    rng = np.random.default_rng()
    gen_fun = rng.standard_normal
    X = gen_fun((K, N ** 2))

    thetas = []

    for k in range(256):
        X_prev = X
        X = op(X)
        X = np.linalg.qr(X.T, 'reduced')[0].T

        if k > 0:
            corr = np.min(np.linalg.svd(X @ X_prev.T)[1])
            theta = np.arccos(corr)
            print('theta = %.15g' % theta)

            thetas.append(theta)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.semilogy(thetas)
    plt.title('N = 128, R = 8, W = 1/8, K = %d' % K)
    plt.ylabel('θ')
    plt.xlabel('t')
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
    print(M)

    M = np.real(eig_compl.T @ op_compl(eig_compl))
    print(M)

    lams_subspace = np.real(lams_subspace)
    lams_subspace = np.sort(lams_subspace)[::-1]

    lams_compl = np.real(lams_compl)
    lams_compl = np.sort(lams_compl)[::-1]

    fname = 'data/large_spectrum.csv'

    util.ensure_dir_exists(fname)

    with open(fname, 'w') as f:
        for k, lam in enumerate(lams_subspace):
            f.write('%d %.15g\n' % (k + 1, lam))

        f.write('\n')

        for k, lam in enumerate(lams_compl):
            f.write('%d %.15g\n' % (k + n_eigs + 1, lam))


if __name__ == '__main__':
    main()

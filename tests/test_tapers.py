import numpy as np
import scipy.linalg

import pytest

from pmte import tapers, util


@pytest.mark.parametrize("N", [4, 5])
@pytest.mark.parametrize("use_fftw", [True, False])
def test_concentration_op_identity_1d(N, use_fftw):
    mask = np.full(N, True)
    op = tapers.concentration_op(mask, W=1, use_fftw=use_fftw)
    A = op(np.eye(N))

    assert np.allclose(A, np.eye(N))


@pytest.mark.parametrize("N, M", [(4, 4), (4, 5)])
@pytest.mark.parametrize("use_fftw", [True, False])
def test_concentration_op_identity_2d(N, M, use_fftw):
    mask = np.full((N, M), True)
    op = tapers.concentration_op(mask, W=1, use_fftw=use_fftw)
    A = op(np.eye(N * M))

    assert np.allclose(A, np.eye(N * M))


@pytest.mark.parametrize("N", [4, 5])
@pytest.mark.parametrize("W", [1 / 2, 1 / 3])
@pytest.mark.parametrize("use_fftw", [True, False])
def test_concentation_op_sinc_1d(N, W, use_fftw):
    mask = np.full(N, True)
    op = tapers.concentration_op(mask, W=W, use_fftw=use_fftw)
    A = op(np.eye(N))

    grid = util.grid((2 * N,), normalized=False)
    sinc = W * np.sinc(W * grid[0])
    A_true = scipy.linalg.toeplitz(sinc[:N])

    assert np.allclose(A, A_true)

    # Test singleton apply.
    assert np.allclose(op(np.eye(N)[:, 0]), A_true[:, 0])


@pytest.mark.parametrize("N, M", [(4, 4), (4, 5)])
@pytest.mark.parametrize("W", [(1 / 2, 1 / 2), (1 / 2, 1 / 3), (1 / 3, 1 / 3)])
@pytest.mark.parametrize("use_fftw", [True, False])
def test_concentation_op_sinc_2d(N, M, W, use_fftw):
    mask = np.full((N, M), True)
    op = tapers.concentration_op(mask, W=W, use_fftw=use_fftw)
    A = op(np.eye(N * M))

    grid1 = util.grid((2 * N,), normalized=False)
    sinc1 = W[0] * np.sinc(W[0] * grid1[0])
    A1_true = scipy.linalg.toeplitz(sinc1[:N])

    grid2 = util.grid((2 * M,), normalized=False)
    sinc2 = W[1] * np.sinc(W[1] * grid2[0])
    A2_true = scipy.linalg.toeplitz(sinc2[:M])

    A_true = np.kron(A1_true, A2_true)

    assert np.allclose(A, A_true)

    # Test singleton apply.
    assert np.allclose(op(np.eye(N * M)[:, 0]), A_true[:, 0])


@pytest.mark.parametrize("N", [7, 8])
@pytest.mark.parametrize("W", [1/16, 1/4, 1/3, 1])
def test_tensor_tapers_1d(N, W):
    h = tapers.tensor_tapers(N, W=W)

    K = round(N * W)

    if W == 1:
        h_true = np.eye(N)
    elif K > 0:
        h_true = scipy.signal.windows.dpss(N, N * W / 2, Kmax=K, norm=2)
    else:
        h_true = 1 / np.sqrt(N) * np.ones((1, N))

    assert np.allclose(h, h_true)


@pytest.mark.parametrize("N, M", [(7, 7), (7, 8), (8, 8)])
@pytest.mark.parametrize("W", [(1/4, 1/4), (1/4, 1/3), (1/3, 1/3)])
def test_tensor_tapers_2d(N, M, W):
    h = tapers.tensor_tapers((N, M), W=W)

    K1 = round(N * W[0])
    K2 = round(M * W[1])

    h1_true = scipy.signal.windows.dpss(N, N * W[0] / 2, Kmax=K1, norm=2)
    h2_true = scipy.signal.windows.dpss(M, M * W[1] / 2, Kmax=K2, norm=2)

    h_true = h1_true[:, None, :, None] * h2_true[None, :, None, :]
    h_true = h_true.reshape((K1 * K2, N, M))

    assert np.allclose(h, h_true)

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

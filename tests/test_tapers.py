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
@pytest.mark.parametrize("W", [1 / 2, (1 / 2, 1 / 2), (1 / 2, 1 / 3), (1 / 3, 1 / 3)])
@pytest.mark.parametrize("use_fftw", [True, False])
def test_concentation_op_sinc_2d(N, M, W, use_fftw):
    mask = np.full((N, M), True)
    op = tapers.concentration_op(mask, W=W, use_fftw=use_fftw)
    A = op(np.eye(N * M))

    if isinstance(W, float):
        W = W * np.ones(2)

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


def test_concentration_op_errors():
    mask = np.full((4,), True)
    W = [0.25, 0.5]

    with pytest.raises(TypeError) as e:
        tapers.concentration_op(mask, W=W)

    assert "must have 1 or d elements" in str(e.value)

    W = 1.1

    with pytest.raises(ValueError) as e:
        tapers.concentration_op(mask, W=W)

    assert "smaller than or equal to 1" in str(e.value)


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
@pytest.mark.parametrize("W", [1/4, (1/4, 1/4), (1/4, 1/3), (1/3, 1/3)])
def test_tensor_tapers_2d(N, M, W):
    h = tapers.tensor_tapers((N, M), W=W)

    if isinstance(W, float):
        W = W * np.ones(2)

    K1 = int(round(N * W[0]))
    K2 = int(round(M * W[1]))

    h1_true = scipy.signal.windows.dpss(N, N * W[0] / 2, Kmax=K1, norm=2)
    h2_true = scipy.signal.windows.dpss(M, M * W[1] / 2, Kmax=K2, norm=2)

    h_true = h1_true[:, None, :, None] * h2_true[None, :, None, :]
    h_true = h_true.reshape((K1 * K2, N, M))

    assert np.allclose(h, h_true)

@pytest.mark.parametrize("N, M", [(7, 7), (7, 8), (8, 8)])
@pytest.mark.parametrize("W", [1/4, (1/4, 1/4), (1/4, 1/3), (1/3, 1/3)])
def test_corner_tapers(N, M, W):
    grid = util.grid((N, M), shifted=True)
    mask = np.hypot(grid[0], grid[1]) > 0.25

    h = tapers.corner_tapers(mask, W=W)

    corner_mask = np.any(np.abs(h) > 0, axis=0)

    N1 = np.argmax(~corner_mask[:, 0])
    N2 = np.argmax(~corner_mask[::-1, 0])
    N3 = np.argmax(~corner_mask[-1, ::-1])
    N4 = np.argmax(~corner_mask[0, ::-1])

    h = h.reshape((h.shape[0], N * M))

    h1 = tapers.tensor_tapers((N1,) * 2, W=W)
    h1 = np.pad(h1, ((0, 0), (0, N - N1), (0, M - N1)))
    h1 = h1.reshape((h1.shape[0], N * M))

    assert np.allclose(h1, (h1 @ h.T) @ h)

    h2 = tapers.tensor_tapers((N2,) * 2, W=W)
    h2 = np.pad(h2, ((0, 0), (N - N2, 0), (0, M - N2)))
    h2 = h2.reshape((h1.shape[0], N * M))

    assert np.allclose(h2, (h2 @ h.T) @ h)

    h3 = tapers.tensor_tapers((N3,) * 2, W=W)
    h3 = np.pad(h3, ((0, 0), (N - N3, 0), (M - N3, 0)))
    h3 = h3.reshape((h3.shape[0], N * M))

    assert np.allclose(h3, (h3 @ h.T) @ h)

    h4 = tapers.tensor_tapers((N4,) * 2, W=W)
    h4 = np.pad(h4, ((0, 0), (0, N - N4), (M - N4, 0)))
    h4 = h4.reshape((h4.shape[0], N * M))

    assert np.allclose(h4, (h4 @ h.T) @ h)


def test_corner_tapers_error():
    mask = np.full((4,), True)

    with pytest.raises(TypeError) as e:
        _ = tapers.corner_tapers(mask, W=1/4)

    assert "for 2D signals" in str(e.value)


@pytest.mark.parametrize("N", [7, 8])
@pytest.mark.parametrize("W", [1/16, 1/4, 1/3, 1])
def test_proxy_tapers_rect_1d(N, W):
    mask = np.full((N,), True)

    h_true = tapers.tensor_tapers((N,), W=W)
    K = h_true.shape[0]

    h = tapers.proxy_tapers(mask, W=W, K=K)

    assert np.isclose(np.max(scipy.linalg.subspace_angles(h, h_true)), 0)


@pytest.mark.parametrize("N, M", [(7, 7), (7, 8), (8, 8)])
@pytest.mark.parametrize("W", [1/4, (1/4, 1/4), (1/4, 1/3), (1/3, 1/3)])
def test_proxy_tapers_rect_2d(N, M, W):
    mask = np.full((N, M), True)

    h_true = tapers.tensor_tapers((N, M), W=W)
    K = h_true.shape[0]

    h = tapers.proxy_tapers(mask, W=W, K=K)

    h = h.reshape(h.shape[0], -1)
    h_true = h_true.reshape(h.shape[0], -1)

    assert np.isclose(np.max(scipy.linalg.subspace_angles(h, h_true)), 0)


@pytest.mark.parametrize("N, M", [(7, 7), (7, 8), (8, 8)])
@pytest.mark.parametrize("W", [1 / 2, (1/4, 1/4), (1/4, 1/3), (1/3, 1/3)])
def test_proxy_tapers_defaults(N, M, W):
    mask = np.full((N, M), True)

    h = tapers.proxy_tapers(mask, W=W)

    if isinstance(W, float):
        W = W * np.ones(2)

    K = int(np.ceil(N * M * W[0] * W[1]))

    assert h.shape[0] == K

import numpy as np

import pytest

from pmte import util


@pytest.mark.parametrize("N", [4, 5])
def test_grid_1d(N):
    grid = util.grid(N)

    grid_true = np.concatenate((np.arange(0, (N + 1) // 2), np.arange(-(N // 2), 0)))
    grid_true = grid_true / N

    assert np.allclose(grid, grid_true)

    grid = util.grid(N, normalized=False)

    grid_true = grid_true * N

    assert np.allclose(grid, grid_true)

    grid = util.grid(N, shifted=True)

    grid_true = np.arange(-(N // 2), (N + 1) // 2)
    grid_true = grid_true / N

    assert np.allclose(grid, grid_true)


@pytest.mark.parametrize("N, M", [(4, 4), (4, 5), (5, 4), (5, 5)])
def test_grid_2d(N, M):
    grid = util.grid((N, M), shifted=True, normalized=False)

    grid1_true = np.arange(-(N // 2), (N + 1) // 2)
    grid2_true = np.arange(-(M // 2), (M + 1) // 2)

    assert np.allclose(grid[0], np.stack((grid1_true,) * M, axis=-1))
    assert np.allclose(grid[1], np.stack((grid2_true,) * N, axis=0))

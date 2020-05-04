import numpy as np
from scipy.signal.windows import dpss

import util
import estimation


def slepian_tapers(n, W):
    V, E = dpss(n, n * W, Kmax=n, norm=2, return_ratios=True)
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


def main():
    N = 128
    R1 = 1 / 16
    R2 = 1 / 3

    N_tensor = int(np.floor(2 * R2 * N))
    tapers, _ = slepian_tapers(N_tensor, R1)

    K = int(np.round(2 * N_tensor * R1))

    tapers = tapers[:, :K]

    tapers = (tapers[:, np.newaxis, :, np.newaxis]
              * tapers[np.newaxis, :, np.newaxis, :])

    tapers = np.reshape(tapers, (N_tensor, N_tensor, -1))

    tentapers = np.zeros((N, N, tapers.shape[2]))

    mid = int(np.ceil((N + 1) / 2) - 1)
    ext1 = int(np.ceil((N_tensor - 1) / 2))
    ext2 = int(np.floor((N_tensor - 1) / 2) + 1)

    tentapers[mid - ext1:mid + ext2, mid - ext1:mid + ext2, :] = tapers.real

    inten = estimation.taper_intensity(tentapers.T).T

    fname = 'data/tentap1.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tentapers[:, :, 0])

    fname = 'data/tentap2.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tentapers[:, :, 1])

    fname = 'data/tentap17.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, tentapers[:, :, 16])

    fname = 'data/teninten.bin'
    util.ensure_dir_exists(fname)
    util.write_gplt_binary_matrix(fname, inten)


if __name__ == '__main__':
    main()

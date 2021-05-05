import sys

import numpy as np

import datahelpers


def main():
    args = sys.argv[1:]

    fname1 = args[0]
    fname2 = args[1]

    relative = ('--relative' in args[2:])

    try:
        X1 = datahelpers.load_float32(fname1)
        X2 = datahelpers.load_float32(fname2)
    except FileNotFoundError:
        print('not found')
        return

    n = np.linalg.norm(X1 - X2)

    if relative:
        n /= np.linalg.norm(X2)

    print(n)


if __name__ == '__main__':
    main()

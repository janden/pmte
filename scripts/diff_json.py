import json
import sys

import numpy as np


def main():
    args = sys.argv[1:]

    fname1 = args[0]
    fname2 = args[1]

    relative = ('--relative' in args[2:])

    try:
        with open(fname1, 'r') as f:
            X1 = json.load(f)
        with open(fname2, 'r') as f:
            X2 = json.load(f)
    except FileNotFoundError:
        print('not found')
        return

    def norm(X):
        return np.linalg.norm([np.linalg.norm(X[k]) for k in X.keys()])

    n = norm({k: X1[k] - X2[k] for k in X1.keys()})

    if relative:
        n /= norm(X2)

    print(n)


if __name__ == '__main__':
    main()

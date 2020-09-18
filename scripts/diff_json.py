import json
import sys

import numpy as np


def main():
    args = sys.argv[1:]

    fname1 = args[0]
    fname2 = args[1]

    try:
        with open(fname1, 'r') as f:
            X1 = json.load(f)
        with open(fname2, 'r') as f:
            X2 = json.load(f)
    except FileNotFoundError:
        print('not found')
        return

    n = np.linalg.norm([np.linalg.norm(X1[k] - X2[k]) for k in X1.keys()])

    print(n)


if __name__ == '__main__':
    main()

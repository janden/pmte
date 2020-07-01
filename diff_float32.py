import sys
import numpy as np

import util


def main():
    args = sys.argv[1:]

    fname1 = args[0]
    fname2 = args[1]

    try:
        X1 = util.load_float32(fname1)
        X2 = util.load_float32(fname2)
    except FileNotFoundError:
        print('not found')
        return

    n = np.linalg.norm(X1 - X2)

    print(n)


if __name__ == '__main__':
    main()

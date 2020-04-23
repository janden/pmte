import numpy as np
import oct2py


def oct_randn():
    oc = oct2py.Oct2Py()
    oc.randn('state', 0)
    gen_fun = lambda sz: np.transpose(oc.randn(*sz[::-1]))
    return gen_fun

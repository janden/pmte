import warnings


def try_import_pyfftw():
    try:
        import pyfftw
        use_fftw = True
    except ImportError:
        warnings.warn('Failed to import pyFFTW. '
                      'Falling back to SciPy FFT.')

        pyfftw = None
        use_fftw = False

    return pyfftw, use_fftw

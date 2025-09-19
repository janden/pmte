# Proxy multitaper estimator for power spectral densities

This repository contains code reproducing the results of

> Andén, J., & Romero, J. L. (2020). Multitaper Estimation on Arbitrary Domains. *SIAM Journal on Imaging Sciences, 13*(3), 1565–1594. https://doi.org/10.1137/19M1278338

## Running

To install the necessary dependencies, run

    pip install -r requirements

The experiments from the paper be executed using

    ./run.sh

and the results are checked against reference results through

    ./check.sh

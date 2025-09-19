#/usr/bin/env bash
set -e

PYTHONPATH="${PYTHONPATH}:${PWD}/src" python3 -m pytest

notebooks=("01-sample_images" \
           "02-spectrum" \
           "03-spectral_window_error" \
           "04-spectral_estimation_error" \
           "05-sample_proxy_tapers" \
           "06-cryo_em_simulation" \
           "07-cryo_em_data")

for notebook in "${notebooks[@]}"
do
    python3 -m nbconvert "notebooks/${notebook}.ipynb" --to python
    echo "[Runnning ${notebook}.py]"
    python3 "notebooks/${notebook}.py"
done

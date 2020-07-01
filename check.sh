ls ref_data/*.csv | xargs -L1 basename | xargs -I{} python3 diff_csv.py data/{} ref_data/{}
ls ref_data/*.bin | xargs -L1 basename | xargs -I{} python3 diff_float32.py data/{} ref_data/{}


ls ref_data/*.csv | xargs -L1 basename | xargs -I{} bash -c "printf %-30s {} && python3 scripts/diff_csv.py data/{} ref_data/{}"
ls ref_data/*.bin | xargs -L1 basename | xargs -I{} bash -c "printf %-30s {} && python3 scripts/diff_float32.py data/{} ref_data/{}"
ls ref_data/*.json | xargs -L1 basename | xargs -I{} bash -c "printf %-30s {} && python3 scripts/diff_json.py data/{} ref_data/{}"

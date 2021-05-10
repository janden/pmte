set -e

ls ref_results/*.csv | xargs -L1 basename | xargs -I{} bash -c "printf %-30s {} && python3 scripts/diff_csv.py results/{} ref_results/{} --relative"
ls ref_results/*.bin | xargs -L1 basename | xargs -I{} bash -c "printf %-30s {} && python3 scripts/diff_float32.py results/{} ref_results/{} --relative"
ls ref_results/*.json | xargs -L1 basename | xargs -I{} bash -c "printf %-30s {} && python3 scripts/diff_json.py results/{} ref_results/{} --relative"

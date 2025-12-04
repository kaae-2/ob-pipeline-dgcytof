#!/usr/bin/env bash
set -euo pipefail

# Run data_preprocessing.py with the requested parameters.
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"

# Prefer the global Windows Python launcher (where your system packages live).
if command -v py >/dev/null 2>&1; then
  python_bin=(py -3)
elif command -v python3 >/dev/null 2>&1; then
  python_bin=(python3)
else
  python_bin=(python)
fi

data_dir="${script_dir}/out/data/preprocessing/data_preprocessing/default"

"${python_bin[@]}" "${script_dir}/dgcytof_cli.py" \
  --name "dgcytof" \
  --output_dir "${script_dir}/out/data/analysis/default/dgcytof" \
  --data.matrix "${data_dir}/data_import.matrix.gz" \
  --data.matrix.train "${data_dir}/data_import.matrix.training.gz" \
  --data.true_labels "${data_dir}/data_import.true_labels.gz" \
  --data.true_labels.train "${data_dir}/data_import.true_labels.training.gz"

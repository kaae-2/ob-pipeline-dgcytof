#!/usr/bin/env bash
set -euo pipefail

# Run data_preprocessing.py with the requested parameters.
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"

python "${script_dir}/dgcytof_cli.py" \
  --name "dgcytof" \
  --output_dir "${script_dir}/out/data/analysis/default/dgcytof" \
  --data.matrix "${script_dir}/out/data/data_preprocessing/default/data_import.matrix.gz" \
  --data.matrix.train "${script_dir}/out/data/data_preprocessing/default/data_import.matrix.training.gz" \
  --data.true_labels "${script_dir}/out/data/data_preprocessing/default/data_import.true_labels.gz" \
  --data.true_labels.train "${script_dir}/out/data/data_preprocessing/default/data_import.true_labels.gz"

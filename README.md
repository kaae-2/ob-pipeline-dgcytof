# DGCyTOF Module

## What this module does

Runs the DGCyTOF model wrapper with Omnibenchmark-compatible I/O.

- CLI: `dgcytof_cli.py`
- Local helper routines: `dgcytof_local.py`
- Local runner: `run_dgcytof.sh`
- Output: `dgcytof_predicted_labels.tar.gz`

## Run locally

```bash
bash models/dgcytof/run_dgcytof.sh
```

## Run as part of benchmark

Configured in `benchmark/Clustering_conda.yml` analysis stage and executed via:

```bash
just benchmark
```

## What `run_dgcytof.sh` needs

- Preprocessing outputs linked at `models/dgcytof/out/data/data_preprocessing/default`
- Python with `torch`, `pandas`, `numpy`, and `scikit-learn`
- Writable output directory `models/dgcytof/out/data/analysis/default/dgcytof`

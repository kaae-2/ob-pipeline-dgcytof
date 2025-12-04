#!/usr/bin/env python
"""
Omnibenchmark runner that mirrors run_agglomerative.py but drives the DGCyTOF
pipeline instead (https://github.com/lijcheng12/DGCyTOF/).

Input/output contract:
* Accepts the same CLI args as run_agglomerative.py (`--data.matrix`,
  `--data.true_labels`, `--output_dir`, `--name`) plus optional training
  inputs (`--data.matrix.train`, `--data.true_labels.train`).
* Supports a single matrix/labels pair or ordered lists of matrices and labels.
* Emits a plain-text file with one predicted label per line (matching the
  formatting of `true_labels`).
"""

import argparse
import gzip
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "PyTorch is required to run DGCyTOF. Install with `pip install torch`."
    ) from exc

try:
    import dgcytof_local as DGCyTOF
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "Missing dgcytof_local module. Ensure dgcytof_local.py is present."
    ) from exc


def _read_first_line(path):
    """Read the first line of a (possibly gzipped) file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        return handle.readline()


def _parse_input_paths(raw_paths):
    """
    Accept a single path, a comma-separated string of paths, or a list/tuple of
    paths and return a flat list with empty entries removed.
    """
    if raw_paths is None:
        return []
    if isinstance(raw_paths, (list, tuple)):
        candidates = list(raw_paths)
    else:
        candidates = [raw_paths]
    paths = []
    for candidate in candidates:
        if candidate is None:
            continue
        for part in str(candidate).split(","):
            trimmed = part.strip()
            if trimmed:
                paths.append(trimmed)
    return paths


def _has_header(first_line):
    """Heuristically decide whether the first line is a header row."""
    tokens = [tok for tok in first_line.replace(",", " ").split() if tok]
    if not tokens:
        return False
    for tok in tokens:
        try:
            float(tok)
        except ValueError:
            return True
    return False


def load_labels(data_file):
    """
    Load labels as 1D array; keeps missing labels as NaN (needed for
    semi-supervised handling in preprocessing).
    """
    opener = gzip.open if data_file.endswith(".gz") else open
    with opener(data_file, "rt") as handle:
        series = pd.read_csv(
            handle,
            header=None,
            comment="#",
            na_values=["", '""', "nan", "NaN"],
            skip_blank_lines=False,
        ).iloc[:, 0]

    try:
        labels = pd.to_numeric(series, errors="coerce").to_numpy()
    except Exception as exc:
        raise ValueError("Invalid data structure, cannot parse labels.") from exc

    if labels.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")
    return labels


def load_dataset(data_file):
    def _read_df(header_opt, engine):
        read_kwargs = {
            "header": header_opt,
            "compression": "infer",
            "dtype": str,  # start as strings to control coercion
        }
        if engine == "python":
            read_kwargs.update({"engine": "python", "sep": None})
        else:
            read_kwargs.update({"engine": "c", "sep": ",", "low_memory": False})
        return pd.read_csv(data_file, **read_kwargs)

    first_line = _read_first_line(data_file)
    has_header = _has_header(first_line)

    dfs_to_try = [0 if has_header else None]
    if has_header:
        dfs_to_try.append(None)  # fallback if header guess was wrong

    last_error = None
    for header_opt in dfs_to_try:
        for engine in ("python", "c"):
            try:
                df = _read_df(header_opt, engine)
            except Exception as exc:
                last_error = exc
                continue

            if header_opt is None:
                df.columns = [f"f{i}" for i in range(df.shape[1])]
            else:
                df.columns = [str(col) for col in df.columns]

            numeric_cols = {}
            dropped = []
            for col in df.columns:
                numeric_series = pd.to_numeric(df[col], errors="coerce")
                if numeric_series.isna().all():
                    dropped.append(col)
                    continue
                fill_value = numeric_series.median()
                numeric_cols[col] = numeric_series.fillna(fill_value)

            if numeric_cols:
                if dropped:
                    sys.stderr.write(
                        f"[dgcytof_cli] Dropped non-numeric columns: {', '.join(map(str, dropped))}\n"
                    )
                numeric_df = pd.DataFrame(numeric_cols)
                numeric_df = numeric_df.dropna(how="all")
                return numeric_df

    raise ValueError(
        f"Data matrix contains no numeric columns; last read error: {last_error}"
    )


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):  # pragma: no cover - passthrough
        return self.model(x)


def run_dgcytof(train_data, train_labels, inference_data, random_state=42):
    """
    Train a small feed-forward network with the DGCyTOF helpers on the provided
    training data and return predicted labels for the inference data
    (1-based to match clustbench).
    """
    if len(train_data) != len(train_labels):
        raise ValueError(
            f"Number of labels ({len(train_labels)}) does not match number of rows in the training data matrix ({len(train_data)})."
        )

    labels_series = pd.to_numeric(pd.Series(train_labels), errors="coerce")
    labels_zero_based = labels_series - 1
    df = train_data.copy()
    df["label"] = labels_zero_based

    # Use only labeled rows for training, but keep the full inference matrix untouched.
    X_data_labeled, y_data, _ = DGCyTOF.preprocessing(df, [])
    if y_data.empty:
        raise ValueError("No labeled rows available after preprocessing.")
    y_data = y_data.astype(int)
    classes = sorted(y_data.unique())
    num_classes = len(classes)
    if num_classes < 2:
        raise ValueError("DGCyTOF requires at least two classes to train.")

    val_size = 0.2
    if len(y_data) * val_size < 1:
        val_size = 0.2 if len(y_data) > 2 else 0.5

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_data_labeled,
            y_data,
            test_size=val_size,
            stratify=y_data,
            random_state=random_state,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_data_labeled,
            y_data,
            test_size=val_size,
            stratify=None,
            random_state=random_state,
        )

    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values.astype(np.int64)),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val.values.astype(np.int64)),
    )

    model_fc = SimpleClassifier(
        input_dim=X_data_labeled.shape[1], num_classes=num_classes
    )

    train_params = {
        "batch_size": min(128, len(train_dataset)),
        "shuffle": True,
        "num_workers": 0,
    }
    val_params = {
        "batch_size": min(10000, len(val_dataset)),
        "shuffle": False,
        "num_workers": 0,
    }

    DGCyTOF.train_model(
        model_fc, train_dataset, max_epochs=20, params_train=train_params
    )
    # Prints accuracy; side effect is fine for benchmarking visibility.
    DGCyTOF.validate_model(model_fc, val_dataset, classes, params_val=val_params)

    model_fc.eval()
    with torch.no_grad():
        full_tensor = torch.tensor(inference_data.values, dtype=torch.float32)
        outputs = model_fc(full_tensor)
        predicted = torch.argmax(outputs, dim=1).cpu().numpy()

    return predicted + 1  # back to 1-based labels


def main():
    parser = argparse.ArgumentParser(description="clustbench DGCyTOF runner")
    parser.add_argument(
        "--data.matrix",
        type=str,
        help="gz-compressed textfile containing the comma-separated data to be clustered.",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--data.true_labels",
        type=str,
        help="gz-compressed textfile with the true labels; used to select a range of ks.",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--data.matrix.train",
        type=str,
        help="Optional training data matrix/matrices (csv/txt). If provided, these are used for training.",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--data.true_labels.train",
        type=str,
        help="Optional training labels corresponding to --data.matrix.train.",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory to store data files.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="name of the dataset",
        default="clustbench",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit(0)

    matrix_paths = _parse_input_paths(getattr(args, "data.matrix"))
    label_paths = _parse_input_paths(getattr(args, "data.true_labels"))
    train_matrix_paths = _parse_input_paths(getattr(args, "data.matrix.train"))
    train_label_paths = _parse_input_paths(getattr(args, "data.true_labels.train"))

    if not matrix_paths:
        raise ValueError("No data matrices provided.")
    if len(matrix_paths) != len(label_paths):
        raise ValueError(
            "The number of data matrices and label files must match "
            f"(got {len(matrix_paths)} matrices vs {len(label_paths)} label files)."
        )

    use_external_training = bool(train_matrix_paths or train_label_paths)
    train_pairs = None
    if use_external_training:
        if not train_matrix_paths or not train_label_paths:
            raise ValueError(
                "Both --data.matrix.train and --data.true_labels.train must be provided to use external training data."
            )
        if len(train_matrix_paths) != len(train_label_paths):
            raise ValueError(
                "The number of training matrices must match the number of training label files."
            )

        if len(train_matrix_paths) not in (1, len(matrix_paths)):
            raise ValueError(
                "Training matrices/labels must be either a single pair (used for all datasets) "
                "or one pair per dataset."
            )

        train_pairs = []
        for idx, (m_path, l_path) in enumerate(zip(train_matrix_paths, train_label_paths)):
            df = load_dataset(m_path)
            labels = load_labels(l_path)
            if len(df) != len(labels):
                min_len = min(len(df), len(labels))
                sys.stderr.write(
                    f"[dgcytof_cli] Training pair {idx} length mismatch ({len(df)} rows vs {len(labels)} labels); "
                    f"truncating to {min_len} to proceed.\n"
                )
                df = df.iloc[:min_len].reset_index(drop=True)
                labels = labels[:min_len]
            train_pairs.append((df, labels))

        ref_cols = train_pairs[0][0].columns
        for idx, (df, _) in enumerate(train_pairs[1:], start=1):
            if not ref_cols.equals(df.columns):
                raise ValueError(
                    f"Training matrices do not share the same columns (mismatch at training index {idx})."
                )

    predictions_by_dataset = []
    for idx, (matrix_path, label_path) in enumerate(zip(matrix_paths, label_paths)):
        truth = load_labels(label_path)
        data_df = load_dataset(matrix_path)

        if len(data_df) != len(truth):
            min_len = min(len(data_df), len(truth))
            sys.stderr.write(
                f"[dgcytof_cli] Dataset {idx} length mismatch ({len(data_df)} rows vs {len(truth)} labels); "
                f"truncating to {min_len} to proceed.\n"
            )
            data_df = data_df.iloc[:min_len].reset_index(drop=True)
            truth = truth[:min_len]

        if train_pairs is not None:
            if len(train_pairs) == 1:
                train_df, train_labels = train_pairs[0]
            else:
                train_df, train_labels = train_pairs[idx]
            common_cols = [col for col in train_df.columns if col in data_df.columns]
            if not common_cols:
                raise ValueError(
                    f"No overlapping feature columns between training data and dataset {matrix_path}."
                )
            if set(common_cols) != set(train_df.columns) or set(common_cols) != set(data_df.columns):
                sys.stderr.write(
                    "[dgcytof_cli] Column mismatch detected; aligning to shared columns only: "
                    f"{len(common_cols)} retained, "
                    f"{len(train_df.columns) - len(common_cols)} dropped from training, "
                    f"{len(data_df.columns) - len(common_cols)} dropped from dataset.\n"
                )
            train_df_aligned = train_df[common_cols]
            data_df_aligned = data_df[common_cols]
            predictions = run_dgcytof(train_df_aligned, train_labels, data_df_aligned)
        else:
            predictions = run_dgcytof(data_df, truth, data_df)

        if len(predictions) != len(truth):
            sys.stderr.write(
                f"[dgcytof_cli] Length mismatch for dataset {matrix_path}: "
                f"predictions={len(predictions)}, truth={len(truth)}, "
                f"data_rows={len(data_df)}, nan_labels={int(pd.isna(truth).sum())}\n"
            )
            raise ValueError(
                f"Predictions and true labels have mismatched lengths for dataset {matrix_path}."
            )

        output_labels = [
            "" if pd.isna(t) else f"{float(p):.1f}"
            for p, t in zip(predictions, truth)
        ]
        predictions_by_dataset.append((matrix_path, output_labels))

    name = args.name
    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)

    if len(predictions_by_dataset) == 1:
        output_path = os.path.join(output_dir, f"{name}_predicted_labels.txt")
        np.savetxt(output_path, np.array(predictions_by_dataset[0][1], dtype=str), fmt="%s")
    else:
        for idx, (matrix_path, output_labels) in enumerate(predictions_by_dataset):
            base = os.path.splitext(os.path.basename(matrix_path))[0] or f"dataset{idx}"
            output_path = os.path.join(
                output_dir, f"{name}_{idx}_{base}_predicted_labels.txt"
            )
            np.savetxt(output_path, np.array(output_labels, dtype=str), fmt="%s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - runtime guard
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.write(f"\nError: {exc}\n")
        sys.exit(1)

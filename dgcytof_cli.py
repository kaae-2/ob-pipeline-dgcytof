#!/usr/bin/env python
"""
Omnibenchmark runner that mirrors run_agglomerative.py but drives the DGCyTOF
pipeline instead (https://github.com/lijcheng12/DGCyTOF/).

Input/output contract:
* Accepts the same CLI args as run_agglomerative.py (`--data.matrix`,
  `--data.true_labels`, `--output_dir`, `--name`).
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
    first_line = _read_first_line(data_file)
    has_header = _has_header(first_line)
    df = pd.read_csv(
        data_file,
        sep=",",
        header=0 if has_header else None,
        compression="infer",
    )
    try:
        df = df.apply(pd.to_numeric)
    except ValueError as exc:
        raise ValueError("Data matrix contains non-numeric values.") from exc

    if not has_header:
        df.columns = [f"f{i}" for i in range(df.shape[1])]
    else:
        df.columns = [str(col) for col in df.columns]
    return df


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


def run_dgcytof(data, labels, random_state=42):
    """
    Train a small feed-forward network with the DGCyTOF helpers and return
    predicted labels for the full dataset (1-based to match clustbench).
    """
    if len(data) != len(labels):
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match number of rows in the data matrix ({len(data)})."
        )

    labels_series = pd.to_numeric(pd.Series(labels), errors="coerce")
    labels_zero_based = labels_series - 1
    df = data.copy()
    df["label"] = labels_zero_based

    # Use only labeled rows for training, but keep the full matrix for inference.
    X_data_labeled, y_data, _ = DGCyTOF.preprocessing(df, [])
    X_full = df.drop(columns=["label"])

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
        full_tensor = torch.tensor(X_full.values, dtype=torch.float32)
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
    )
    parser.add_argument(
        "--data.true_labels",
        type=str,
        help="gz-compressed textfile with the true labels; used to select a range of ks.",
        required=True,
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

    truth = load_labels(getattr(args, "data.true_labels"))
    data_matrix = getattr(args, "data.matrix")
    data_df = load_dataset(data_matrix)
    predictions = run_dgcytof(data_df, truth)

    if len(predictions) != len(truth):
        sys.stderr.write(
            f"[dgcytof_cli] Length mismatch: predictions={len(predictions)}, "
            f"truth={len(truth)}, data_rows={len(data_df)}, "
            f"nan_labels={int(pd.isna(truth).sum())}\n"
        )
        raise ValueError("Predictions and true labels have mismatched lengths.")

    name = args.name
    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_predicted_labels.txt")
    output_labels = [
        "" if pd.isna(t) else f"{float(p):.1f}"
        for p, t in zip(predictions, truth)
    ]
    np.savetxt(output_path, np.array(output_labels, dtype=str), fmt="%s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - runtime guard
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.write(f"\nError: {exc}\n")
        sys.exit(1)

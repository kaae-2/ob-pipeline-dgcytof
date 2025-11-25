#!/usr/bin/env python
"""
Omnibenchmark runner that mirrors run_agglomerative.py but drives the DGCyTOF
pipeline instead (https://github.com/lijcheng12/DGCyTOF/).

Input/output contract:
* Accepts the same CLI args as run_agglomerative.py (`--data.matrix`,
  `--data.true_labels`, `--output_dir`, `--name`).
* Writes a gz-compressed CSV with a single column of predicted labels.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Use the vendored DGCyTOF copy under ./dgcytof/DGCyTOF_Package
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DGCYTOF_PATH = os.path.join(HERE, "dgcytof", "DGCyTOF_Package")
if LOCAL_DGCYTOF_PATH not in sys.path:
    sys.path.insert(0, LOCAL_DGCYTOF_PATH)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "PyTorch is required to run DGCyTOF. Install with `pip install torch`."
    ) from exc

try:
    import DGCyTOF
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "Missing DGCyTOF package. Ensure ./dgcytof/DGCyTOF_Package is present "
        "or install with `pip install ./dgcytof/DGCyTOF_Package`."
    ) from exc


def load_labels(data_file):
    data = np.loadtxt(data_file, ndmin=1)
    if data.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")
    return data


def load_dataset(data_file):
    data = np.loadtxt(data_file, ndmin=2)
    if data.ndim != 2:
        raise ValueError("Invalid data structure, not a 2D matrix?")
    return data


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, **kwargs):
        # Some environments/wrappers may pass unused keywords like `inputs`/`outputs`;
        # ignore them to avoid torch.nn.Module.__init__ TypeError.
        kwargs.pop("inputs", None)
        kwargs.pop("outputs", None)
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
    labels_zero_based = labels.astype(int) - 1
    feature_cols = [f"f{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=feature_cols)
    df["label"] = labels_zero_based

    X_data_labeled, y_data, _ = DGCyTOF.preprocessing(df, [])
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

    model_fc = SimpleClassifier(input_dim=data.shape[1], num_classes=num_classes)

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
        full_tensor = torch.tensor(X_data_labeled.values, dtype=torch.float32)
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
    data = getattr(args, "data.matrix")
    predictions = run_dgcytof(load_dataset(data), truth)

    header = np.array(["prediction"]).reshape(1, 1)
    curr = np.append(header, predictions.astype(str).reshape(-1, 1), axis=0)

    name = args.name
    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(
        os.path.join(output_dir, f"{name}.labels.gz"),
        curr,
        fmt="%s",
        delimiter=",",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - runtime guard
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.write(f"\nError: {exc}\n")
        sys.exit(1)

#!/usr/bin/env python
"""
Omnibenchmark runner that mirrors run_agglomerative.py but drives the DGCyTOF
pipeline instead (https://github.com/lijcheng12/DGCyTOF/).

Input/output contract:
* Accepts training and test inputs (`--data.train_matrix`,
  `--data.train_labels`, `--data.test_matrix`, `--output_dir`, `--name`).
* Emits a tar.gz of per-sample prediction CSVs for the test set.

"""

import argparse
import contextlib
import gzip
import io
import os
import re
import sys
import tarfile
import tempfile
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime guard
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = exc

try:
    import dgcytof_local as DGCyTOF
    DGCYTOF_AVAILABLE = True
    DGCYTOF_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime guard
    DGCyTOF = None  # type: ignore[assignment]
    DGCYTOF_AVAILABLE = False
    DGCYTOF_IMPORT_ERROR = exc


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


def _normalize_labels(series: pd.Series) -> np.ndarray:
    numeric = pd.Series(pd.to_numeric(series, errors="coerce"))
    if numeric.notna().any():
        numeric = numeric.mask(numeric == 0)
        labels = numeric.to_numpy()
    else:
        # Map textual labels to integers. Treat empty strings and 'unlabeled' as missing.
        str_series = series.astype(str).str.strip()
        mask_valid = ~(
            str_series.isna()
            | str_series.eq("")
            | str_series.str.lower().eq("unlabeled")
        )
        unique_labels = sorted(str_series[mask_valid].unique())
        if not unique_labels:
            # No usable labels found
            labels = numeric.to_numpy()
        else:
            mapping = {lab: i + 1 for i, lab in enumerate(unique_labels)}
            mapped = str_series.map(mapping).astype(float)
            # Unmapped entries become NaN
            mapped[~mask_valid] = float("nan")
            labels = mapped.to_numpy()

    if getattr(labels, "ndim", None) != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")
    return labels


def load_labels(data_file):
    """
    Load labels as 1D numeric array. If the labels are textual (e.g. gate names),
    map unique non-empty, non-'unlabeled' strings to integer class ids starting at 1.
    Keep missing/unlabeled as NaN to allow semi-supervised handling downstream.
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
    return _normalize_labels(series)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df_numeric = df.apply(pd.to_numeric, axis=0)
    except Exception:
        df_numeric = df.apply(pd.to_numeric, axis=0, errors="coerce")
        df_numeric = df_numeric.dropna(how="all")
        if df_numeric.empty:
            raise ValueError("Data matrix contains non-numeric values.")
    if isinstance(df_numeric, pd.Series):
        return df_numeric.to_frame()
    return df_numeric


def load_dataset(data_file):
    first_line = _read_first_line(data_file)
    has_header = _has_header(first_line)
    df = pd.read_csv(
        data_file,
        sep=",",
        header=0 if has_header else None,
        compression="infer",
    )
    df = _coerce_numeric(df)

    if not has_header:
        df.columns = [f"f{i}" for i in range(df.shape[1])]
    else:
        df.columns = [str(col) for col in df.columns]
    return df


def _load_single_csv_from_tar(data_file: str) -> Tuple[str, pd.DataFrame]:
    if not tarfile.is_tarfile(data_file):
        return os.path.basename(data_file), load_dataset(data_file)

    with tarfile.open(data_file, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        if not members:
            raise ValueError(f"No files found in archive: {data_file}")
        member = sorted(members, key=lambda m: m.name)[0]
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise ValueError(f"Unable to read {member.name} from {data_file}")
        data = file_obj.read()
        if member.name.endswith(".gz"):
            data = gzip.decompress(data)
        df = pd.read_csv(io.BytesIO(data), header=None)
        df = _coerce_numeric(df)
        df.columns = [f"f{i}" for i in range(df.shape[1])]
        return member.name, df


def _load_labels_from_tar(data_file: str) -> np.ndarray:
    if not tarfile.is_tarfile(data_file):
        return load_labels(data_file)

    with tarfile.open(data_file, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        if not members:
            raise ValueError(f"No files found in archive: {data_file}")
        member = sorted(members, key=lambda m: m.name)[0]
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise ValueError(f"Unable to read {member.name} from {data_file}")
        data = file_obj.read()
        if member.name.endswith(".gz"):
            data = gzip.decompress(data)
        series = pd.read_csv(
            io.BytesIO(data),
            header=None,
            comment="#",
            na_values=["", '""', "nan", "NaN"],
            skip_blank_lines=False,
        ).iloc[:, 0]
        return _normalize_labels(series)


def _extract_sample_number(sample_name: str) -> Optional[str]:
    base = os.path.basename(sample_name)
    for suffix in ('.csv.gz', '.labels.gz', '.label.gz', '.csv', '.gz'):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    match = re.search(r"(\d+)(?!.*\d)", base)
    if match:
        return match.group(1)
    return None


def load_test_samples(data_file: str) -> List[Tuple[str, pd.DataFrame, Optional[str]]]:
    if not tarfile.is_tarfile(data_file):
        sample_name = os.path.basename(data_file)
        return [
            (sample_name, load_dataset(data_file), _extract_sample_number(sample_name))
        ]

    samples: List[Tuple[str, pd.DataFrame, Optional[str]]] = []
    with tarfile.open(data_file, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for member in members:
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            data = file_obj.read()
            if member.name.endswith(".gz"):
                data = gzip.decompress(data)
            df = pd.read_csv(io.BytesIO(data), header=None)
            df = _coerce_numeric(df)
            df.columns = [f"f{i}" for i in range(df.shape[1])]
            samples.append((member.name, df, _extract_sample_number(member.name)))

    if not samples:
        return [("empty", pd.DataFrame(), None)]
    return samples


def train_dgcytof(train_data, train_labels, random_state=42):
    """
    Train a small feed-forward network with the DGCyTOF helpers and return
    predicted labels for the full dataset (1-based to match clustbench).
    """
    if len(train_data) != len(train_labels):
        raise ValueError(
            f"Number of labels ({len(train_labels)}) does not match number of rows in the data matrix ({len(train_data)})."
        )

    labels_series = pd.Series(train_labels)
    labels_numeric = pd.Series(pd.to_numeric(labels_series, errors="coerce"))
    labels_array = labels_numeric.to_numpy(dtype=float)
    labels_zero_based = labels_array.astype(float)
    labeled_mask = np.isfinite(labels_zero_based) & (labels_zero_based > 0)
    if not np.any(labeled_mask):
        raise ValueError("No labeled rows available after preprocessing.")

    classes = sorted({int(value) for value in labels_zero_based[labeled_mask]})
    num_classes = len(classes)
    if num_classes < 2:
        raise ValueError("DGCyTOF requires at least two classes to train.")

    if not TORCH_AVAILABLE or not DGCYTOF_AVAILABLE:
        reasons = []
        if not TORCH_AVAILABLE and TORCH_IMPORT_ERROR is not None:
            reasons.append(f"PyTorch unavailable ({TORCH_IMPORT_ERROR})")
        if not DGCYTOF_AVAILABLE and DGCYTOF_IMPORT_ERROR is not None:
            reasons.append(f"dgcytof_local unavailable ({DGCYTOF_IMPORT_ERROR})")
        print(
            "DGCyTOF: using sklearn MLP fallback because " + "; ".join(reasons),
            file=sys.stderr,
            flush=True,
        )
        classifier = make_pipeline(
            SimpleImputer(strategy="median"),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=random_state,
            ),
        )
        classifier.fit(train_data.loc[labeled_mask].to_numpy(), labels_zero_based[labeled_mask].astype(int))
        return classifier, None

    df = train_data.copy()
    df["label"] = labels_zero_based

    assert DGCyTOF is not None
    assert torch is not None
    assert nn is not None
    assert TensorDataset is not None
    torch_nn = cast(Any, nn)

    # Use only labeled rows for training, but keep the full matrix for inference.
    X_data_labeled, y_data, _ = DGCyTOF.preprocessing(df, [])
    if y_data.empty:
        raise ValueError("No labeled rows available after preprocessing.")
    y_data = y_data.astype(int)
    classes = sorted(y_data.unique())
    label_map = {label: idx for idx, label in enumerate(classes)}
    y_data = y_data.map(label_map).astype(int)
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

    X_train_arr = np.asarray(X_train)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    X_val_arr = np.asarray(X_val)
    y_val_arr = np.asarray(y_val, dtype=np.int64)
    train_dataset = TensorDataset(
        torch.tensor(X_train_arr, dtype=torch.float32),
        torch.tensor(y_train_arr),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_arr, dtype=torch.float32),
        torch.tensor(y_val_arr),
    )

    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.model = torch_nn.Sequential(
                torch_nn.Linear(input_dim, 128),
                torch_nn.ReLU(),
                torch_nn.Dropout(0.1),
                torch_nn.Linear(128, 64),
                torch_nn.ReLU(),
                torch_nn.Linear(64, num_classes),
            )

        def forward(self, x):  # pragma: no cover - passthrough
            return self.model(x)

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

    with contextlib.redirect_stdout(io.StringIO()):
        DGCyTOF.train_model(
            model_fc, train_dataset, max_epochs=20, params_train=train_params
        )
        DGCyTOF.validate_model(model_fc, val_dataset, classes, params_val=val_params)

    model_fc.eval()
    return model_fc, classes


def predict_dgcytof(
    model, data: pd.DataFrame, classes: Optional[List[int]]
) -> np.ndarray:
    if hasattr(model, "predict") and classes is None:
        predicted = model.predict(data.to_numpy())
        return np.asarray(predicted, dtype=int)

    assert torch is not None
    assert classes is not None
    model.eval()
    with torch.no_grad():
        full_tensor = torch.tensor(data.values, dtype=torch.float32)
        outputs = model(full_tensor)
        predicted = torch.argmax(outputs, dim=1).cpu().numpy()

    return np.asarray([classes[idx] for idx in predicted], dtype=int)


def main():
    parser = argparse.ArgumentParser(description="clustbench DGCyTOF runner")
    parser.add_argument(
        "--data.train_matrix",
        type=str,
        help="gz-compressed CSV containing training data.",
        required=True,
    )
    parser.add_argument(
        "--data.train_labels",
        type=str,
        help="gz-compressed CSV containing training labels.",
        required=True,
    )
    parser.add_argument(
        "--data.test_matrix",
        type=str,
        help="gz-compressed CSV containing test data.",
        required=True,
    )
    parser.add_argument(
        "--data.metadata",
        type=str,
        help="metadata JSON.gz path (accepted but unused).",
        required=False,
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

    name = args.name
    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)

    print("DGCyTOF: loading training data", flush=True)
    _train_data_name, train_data = _load_single_csv_from_tar(
        getattr(args, "data.train_matrix")
    )
    train_labels = _load_labels_from_tar(getattr(args, "data.train_labels"))
    print("DGCyTOF: loading test data", flush=True)
    test_samples = load_test_samples(getattr(args, "data.test_matrix"))

    print("DGCyTOF: training model", flush=True)
    model, classes = train_dgcytof(train_data, train_labels)

    output_path = os.path.join(output_dir, f"{name}_predicted_labels.tar.gz")
    if os.path.islink(output_path):
        os.unlink(output_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_files: List[str] = []
        print("DGCyTOF: generating predictions", flush=True)
        for sample_name, sample_df, sample_number in test_samples:
            predictions = predict_dgcytof(model, sample_df, classes)
            output_labels = ["" if pd.isna(p) else f"{int(p)}" for p in predictions]
            if sample_number is None:
                sample_number = str(len(output_files) + 1)
            safe_name = f"{name}-prediction-{sample_number}.csv"
            file_path = os.path.join(tmpdir, safe_name)
            with open(file_path, "wt") as handle:
                pd.Series(output_labels).to_csv(handle, index=False, header=False)
            output_files.append(file_path)

        with tarfile.open(output_path, "w:gz") as tar:
            for path in output_files:
                tar.add(path, arcname=os.path.basename(path))
    print("DGCyTOF: finished", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - runtime guard
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.write(f"\nError: {exc}\n")
        sys.exit(1)

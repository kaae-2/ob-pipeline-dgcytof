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
import hashlib
import io
import json
import os
import platform
import re
import sys
import tarfile
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

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

VENDORED_PACKAGE_ROOT = (
    Path(__file__).resolve().parent / 'dgcytof' / 'DGCyTOF_Package'
)
VENDORED_PACKAGE_SOURCE = VENDORED_PACKAGE_ROOT / 'DGCyTOF' / '__init__.py'
sys.path.insert(0, str(VENDORED_PACKAGE_ROOT))
try:
    import DGCyTOF  # noqa: E402
except Exception as exc:
    raise ImportError(
        f'genuine vendored DGCyTOF package unavailable at {VENDORED_PACKAGE_ROOT}'
    ) from exc


def verify_vendored_package(package):
    package_path = Path(package.__file__).resolve()
    if package_path != VENDORED_PACKAGE_SOURCE.resolve():
        raise ImportError(
            f'DGCyTOF resolved outside the vendored package: {package_path}'
        )
    return package_path


verify_vendored_package(DGCyTOF)
DGCYTOF_PROVENANCE = {
    'package_path': str(Path(DGCyTOF.__file__).resolve()),
    'source_sha256': hashlib.sha256(VENDORED_PACKAGE_SOURCE.read_bytes()).hexdigest(),
    'version': re.search(
        r'version="([^"]+)"',
        (VENDORED_PACKAGE_ROOT / 'setup.py').read_text(encoding='utf-8'),
    ).group(1),
}
DGCYTOF_AVAILABLE = True
DGCYTOF_IMPORT_ERROR = None


class Model_fc(nn.Module):
    """Notebook DGCyTOF fully connected classifier."""

    def __init__(self, input_features, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_labels, bias=True)

    def forward(self, inputs):
        inputs = nn.functional.relu(self.fc1(inputs))
        inputs = nn.functional.relu(self.fc2(inputs))
        inputs = nn.functional.relu(self.fc3(inputs))
        return self.out(inputs)


class _LazySpearman:
    def __init__(self, transposed_cells):
        cells = np.asarray(transposed_cells).T
        ranked = rankdata(cells, axis=1)
        centered = ranked - ranked.mean(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.normalized = centered / np.linalg.norm(
                centered, axis=1, keepdims=True
            )

    def active_learning(self, correct_size):
        correct = self.normalized[:correct_size]
        if correct_size < 2:
            average_correct = float('nan')
        else:
            summed = correct.sum(axis=0)
            pair_sum = (
                np.dot(summed, summed)
                - np.einsum('ij,ij->', correct, correct)
            ) / 2
            average_correct = pair_sum / (
                correct_size * (correct_size - 1) / 2
            )

        candidates = self.normalized[correct_size + 1 :]
        if len(candidates) == 0:
            return [], [], average_correct
        correlations = candidates @ correct.mean(axis=0)
        selected = np.flatnonzero(correlations > average_correct)
        return (
            (selected + 1).tolist(),
            correlations[selected].tolist(),
            average_correct,
        )


def calibrate_with_indices(model, data, classes, validation_results):
    """Call vendored calibration while retaining each input row position."""
    correct_probabilities = [
        np.max(nn.functional.softmax(out, dim=0).data.numpy())
        for predicted, label, out in validation_results
        if predicted == label
    ]
    if not correct_probabilities:
        raise RuntimeError('DGCyTOF calibration has no correct validation results.')

    inputs = torch.tensor(data.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs)
        predicted_indices = torch.argmax(outputs, dim=1).cpu().numpy()
        probabilities = (
            nn.functional.softmax(outputs, dim=1).max(dim=1).values.cpu().numpy()
        )
    threshold = float(min(correct_probabilities))
    rounded_probabilities = np.asarray([round(value, 4) for value in probabilities])
    uncertain_positions = np.flatnonzero(rounded_probabilities <= threshold)

    active_results = []
    original_spearmanr = DGCyTOF.spearmanr
    original_active_learning = DGCyTOF.active_learning_index

    def lazy_spearmanr(values):
        return _LazySpearman(values), None

    def indexed_active_learning(test_correct_list, rho, indices=True):
        if not indices:
            return rho.active_learning(len(test_correct_list))[2]
        result = rho.active_learning(len(test_correct_list))
        active_results.append(result)
        return result

    try:
        DGCyTOF.spearmanr = lazy_spearmanr
        DGCyTOF.active_learning_index = indexed_active_learning
        vendored_unresolved = DGCyTOF.calibrate_data(
            model,
            inputs,
            classes,
            validation_results,
            data,
        )
    finally:
        DGCyTOF.spearmanr = original_spearmanr
        DGCyTOF.active_learning_index = original_active_learning

    if len(active_results) != len(classes):
        raise RuntimeError(
            'DGCyTOF calibration did not evaluate every known class.'
        )

    correlations = np.zeros((len(uncertain_positions), len(classes)))
    for class_index, (indices, values, _average) in enumerate(active_results):
        correlations[np.asarray(indices, dtype=int), class_index] = values
    promoted_mask = (correlations != 0).any(axis=1)
    promoted_classes = correlations[promoted_mask].argmax(axis=1)
    unresolved_positions = uncertain_positions[~promoted_mask]

    predictions = np.asarray(
        [classes[index] for index in predicted_indices], dtype=int
    )
    predictions[uncertain_positions] = 0
    predictions[uncertain_positions[promoted_mask]] = np.asarray(classes)[
        promoted_classes
    ]

    vendored_rows = Counter(
        map(tuple, np.asarray(vendored_unresolved, dtype=np.float32))
    )
    indexed_rows = Counter(
        map(tuple, inputs[unresolved_positions].cpu().numpy())
    )
    if vendored_rows != indexed_rows:
        raise RuntimeError(
            'Index-preserving calibration disagrees with vendored unresolved rows.'
        )

    return {
        'predictions': predictions,
        'unresolved_positions': unresolved_positions.tolist(),
        'unresolved_index': data.index[unresolved_positions].tolist(),
        'threshold': threshold,
        'confident_count': int(len(data) - len(uncertain_positions)),
        'initial_uncertain_count': int(len(uncertain_positions)),
        'promoted_count': int(promoted_mask.sum()),
        'unresolved_count': int(len(unresolved_positions)),
    }


def classify_sample(model, data, classes, validation_results):
    calibration = calibrate_with_indices(
        model, data, classes, validation_results
    )
    unresolved_positions = calibration['unresolved_positions']
    if len(unresolved_positions) >= 10:
        unresolved_data = data.iloc[unresolved_positions].to_numpy(
            dtype=np.float32
        )
        try:
            (
                _embedding,
                cluster_labels,
                new_subtypes,
                figure,
            ) = DGCyTOF.dimensionality_reduction_and_clustering(
                unresolved_data
            )
        finally:
            DGCyTOF.plt.close('all')
        cluster_counts = Counter(int(label) for label in cluster_labels)
        clustering = {
            'invoked': True,
            'input_count': len(unresolved_positions),
            'cluster_counts': {
                str(label): count
                for label, count in sorted(cluster_counts.items())
            },
            'new_subtypes': list(new_subtypes),
        }
    elif unresolved_positions:
        clustering = {
            'invoked': False,
            'input_count': len(unresolved_positions),
            'cluster_counts': {},
            'new_subtypes': [],
            'reason': (
                'fewer than 10 unresolved cells; subtype clustering requires '
                'at least 10'
            ),
        }
    else:
        clustering = {
            'invoked': False,
            'input_count': 0,
            'cluster_counts': {},
            'new_subtypes': [],
            'reason': 'no cells remained unresolved after calibration',
        }

    predictions = calibration['predictions']
    return {
        'predictions': predictions,
        'calibration': {
            key: value
            for key, value in calibration.items()
            if key not in {'predictions', 'unresolved_positions', 'unresolved_index'}
        },
        'clustering': clustering,
        'rejection_count': int(np.count_nonzero(predictions == 0)),
    }


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
        raise RuntimeError(
            "DGCyTOF dependencies are unavailable: " + "; ".join(reasons)
        )

    df = train_data.copy()
    df["label"] = labels_zero_based

    assert DGCyTOF is not None
    assert torch is not None
    assert nn is not None
    assert TensorDataset is not None
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

    model_fc = Model_fc(
        input_features=X_data_labeled.shape[1], num_labels=num_classes
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
        validation_results = DGCyTOF.validate_model(
            model_fc, val_dataset, classes, params_val=val_params
        )

    model_fc.eval()
    return model_fc, classes, validation_results


def predict_dgcytof(
    model, data: pd.DataFrame, classes: Optional[List[int]]
) -> np.ndarray:
    assert torch is not None
    assert classes is not None
    model.eval()
    with torch.no_grad():
        full_tensor = torch.tensor(data.values, dtype=torch.float32)
        outputs = model(full_tensor)
        predicted = torch.argmax(outputs, dim=1).cpu().numpy()

    return np.asarray([classes[idx] for idx in predicted], dtype=int)


def write_prediction_archive(output_files, output_path):
    output = Path(output_path)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f'.{output.name}.',
        suffix='.tmp',
        delete=False,
    ) as pending_handle:
        pending = Path(pending_handle.name)
    try:
        with tarfile.open(pending, 'w:gz') as tar:
            for path in output_files:
                tar.add(path, arcname=os.path.basename(path))
        os.replace(pending, output)
    finally:
        pending.unlink(missing_ok=True)


def load_metadata(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        metadata = json.load(handle)
    id_to_label = metadata.get('labels', {}).get('id_to_label')
    if not isinstance(id_to_label, dict) or not id_to_label:
        raise ValueError('Metadata does not define labels.id_to_label.')
    metadata_ids = sorted(int(value) for value in id_to_label)
    if any(value <= 0 for value in metadata_ids):
        raise ValueError('Metadata population IDs must be positive integers.')
    return metadata, metadata_ids


def dependency_versions():
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torchvision': version('torchvision'),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scipy': version('scipy'),
        'scikit-learn': version('scikit-learn'),
        'matplotlib': version('matplotlib'),
        'umap-learn': version('umap-learn'),
        'hdbscan': version('hdbscan'),
    }


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
        help="metadata JSON.gz defining benchmark population IDs.",
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
    parser.add_argument("--seed", type=int, default=42)

    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit(0)

    name = args.name
    np.random.seed(args.seed)
    if TORCH_AVAILABLE:
        assert torch is not None
        torch.manual_seed(args.seed)
    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_predicted_labels.tar.gz")
    provenance_path = os.path.join(output_dir, f'{name}_dgcytof_provenance.json')
    for stale_path in (output_path, provenance_path):
        if os.path.lexists(stale_path):
            os.unlink(stale_path)

    print("DGCyTOF: loading training data", flush=True)
    _train_data_name, train_data = _load_single_csv_from_tar(
        getattr(args, "data.train_matrix")
    )
    train_labels = _load_labels_from_tar(getattr(args, "data.train_labels"))
    _metadata, metadata_ids = load_metadata(getattr(args, 'data.metadata'))
    print("DGCyTOF: loading test data", flush=True)
    test_samples = load_test_samples(getattr(args, "data.test_matrix"))

    print("DGCyTOF: training model", flush=True)
    model, classes, validation_results = train_dgcytof(
        train_data, train_labels, random_state=args.seed
    )
    if not set(classes).issubset(metadata_ids):
        raise ValueError(
            f'Training classes {classes} are outside metadata IDs {metadata_ids}.'
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_files: List[str] = []
        sample_provenance = []
        print("DGCyTOF: generating predictions", flush=True)
        for sample_name, sample_df, sample_number in test_samples:
            result = classify_sample(
                model, sample_df, classes, validation_results
            )
            predictions = result['predictions']
            invalid = set(np.unique(predictions)) - {0, *metadata_ids}
            if invalid:
                raise RuntimeError(
                    f'Out-of-domain predictions for {sample_name}: {sorted(invalid)}'
                )
            output_labels = ["" if pd.isna(p) else f"{int(p)}" for p in predictions]
            if sample_number is None:
                sample_number = str(len(output_files) + 1)
            safe_name = f"{name}-prediction-{sample_number}.csv"
            file_path = os.path.join(tmpdir, safe_name)
            with open(file_path, "wt") as handle:
                pd.Series(output_labels).to_csv(handle, index=False, header=False)
            output_files.append(file_path)
            sample_provenance.append(
                {
                    'sample_name': sample_name,
                    'sample_number': sample_number,
                    'row_count': len(sample_df),
                    'calibration': result['calibration'],
                    'clustering': result['clustering'],
                    'rejection_count': result['rejection_count'],
                }
            )

        write_prediction_archive(output_files, output_path)

    provenance = {
        'package': DGCYTOF_PROVENANCE,
        'architecture': [
            'Linear(input,128)',
            'ReLU',
            'Linear(128,64)',
            'ReLU',
            'Linear(64,32)',
            'ReLU',
            'Linear(32,classes)',
        ],
        'dependencies': dependency_versions(),
        'seed': args.seed,
        'metadata_ids': metadata_ids,
        'authoritative_stages': [
            'DGCyTOF.preprocessing',
            'DGCyTOF.train_model',
            'DGCyTOF.validate_model',
            'DGCyTOF.calibrate_data',
            'DGCyTOF.dimensionality_reduction_and_clustering',
        ],
        'calibration_adapter': (
            'index-preserving lazy Spearman implementation with runtime '
            'multiset agreement check against vendored calibrate_data output'
        ),
        'fallback_used': False,
        'samples': sample_provenance,
        'totals': {
            'rows': sum(item['row_count'] for item in sample_provenance),
            'calibration_initial_uncertain': sum(
                item['calibration']['initial_uncertain_count']
                for item in sample_provenance
            ),
            'calibration_promoted': sum(
                item['calibration']['promoted_count']
                for item in sample_provenance
            ),
            'clustering_input': sum(
                item['clustering']['input_count'] for item in sample_provenance
            ),
            'rejections': sum(
                item['rejection_count'] for item in sample_provenance
            ),
        },
    }
    with open(provenance_path, 'wt', encoding='utf-8') as handle:
        json.dump(provenance, handle, indent=2)
        handle.write('\n')
    print(json.dumps(provenance, sort_keys=True), flush=True)
    print("DGCyTOF: finished", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - runtime guard
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.write(f"\nError: {exc}\n")
        sys.exit(1)

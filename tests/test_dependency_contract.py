import importlib.util
import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


MODULE_PATH = Path(__file__).parents[1] / 'dgcytof_cli.py'
SPEC = importlib.util.spec_from_file_location('dgcytof_cli', MODULE_PATH)
dgcytof_cli = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(dgcytof_cli)


class FixtureModel(torch.nn.Module):
    def forward(self, inputs):
        return torch.column_stack((inputs[:, 0], -inputs[:, 0]))


def calibration_fixture():
    data = pd.DataFrame(
        [
            [3.0, 2.0, 1.0, 0.0],
            [3.0, 0.0, 2.0, 1.0],
            [3.0, 1.0, 0.0, 2.0],
            [-3.0, -2.0, -1.0, 0.0],
            [-3.0, 0.0, -2.0, -1.0],
            [-3.0, -1.0, 0.0, -2.0],
            [0.1, 3.0, 2.0, 1.0],
            [-0.1, -3.0, -2.0, -1.0],
        ],
        columns=['a', 'b', 'c', 'd'],
    )
    validation_results = [
        (torch.tensor(0), torch.tensor(0), torch.tensor([0.7, -0.7])),
        (torch.tensor(1), torch.tensor(1), torch.tensor([-0.7, 0.7])),
    ]
    return FixtureModel(), data, validation_results


class DependencyContractTests(unittest.TestCase):
    def test_imports_the_vendored_dgcytof_package_with_source_provenance(self):
        expected = (
            MODULE_PATH.parent
            / 'dgcytof'
            / 'DGCyTOF_Package'
            / 'DGCyTOF'
            / '__init__.py'
        ).resolve()

        self.assertEqual(dgcytof_cli.DGCyTOF.__name__, 'DGCyTOF')
        self.assertEqual(Path(dgcytof_cli.DGCyTOF.__file__).resolve(), expected)
        self.assertEqual(
            dgcytof_cli.DGCYTOF_PROVENANCE['source_sha256'],
            hashlib.sha256(expected.read_bytes()).hexdigest(),
        )

    def test_rejects_local_helper_provenance(self):
        local_helper = SimpleNamespace(
            __file__=str(MODULE_PATH.parent / 'dgcytof_local.py')
        )

        with self.assertRaisesRegex(ImportError, 'outside the vendored package'):
            dgcytof_cli.verify_vendored_package(local_helper)

    def test_model_fc_matches_the_notebook_architecture(self):
        model = dgcytof_cli.Model_fc(input_features=8, num_labels=3)
        linear_shapes = [
            (layer.in_features, layer.out_features)
            for layer in model.modules()
            if isinstance(layer, torch.nn.Linear)
        ]

        self.assertEqual(
            linear_shapes,
            [(8, 128), (128, 64), (64, 32), (32, 3)],
        )
        self.assertFalse(
            any(isinstance(layer, torch.nn.Dropout) for layer in model.modules())
        )
        self.assertEqual(tuple(model(torch.zeros(2, 8)).shape), (2, 3))

    def test_indexed_calibration_matches_vendored_unresolved_rows(self):
        model, data, validation_results = calibration_fixture()
        reference = dgcytof_cli.DGCyTOF.calibrate_data(
            model,
            torch.tensor(data.values, dtype=torch.float32),
            [1, 2],
            validation_results,
            data,
        )

        result = dgcytof_cli.calibrate_with_indices(
            model,
            data,
            [1, 2],
            validation_results,
        )
        reference_rows = Counter(map(tuple, np.asarray(reference)))
        indexed_rows = Counter(
            map(
                tuple,
                data.iloc[result['unresolved_positions']].to_numpy(
                    dtype=np.float32
                ),
            )
        )

        self.assertEqual(indexed_rows, reference_rows)
        self.assertEqual(len(result['unresolved_positions']), len(reference))

    def test_calibration_preserves_duplicate_row_identity_and_order(self):
        model, data, validation_results = calibration_fixture()
        data.iloc[7] = data.iloc[6]
        data.index = [10, 11, 12, 20, 21, 22, 501, 502]

        result = dgcytof_cli.calibrate_with_indices(
            model,
            data,
            [1, 2],
            validation_results,
        )

        self.assertEqual(result['unresolved_index'], [501])
        self.assertEqual(
            result['predictions'].tolist(),
            [1, 1, 1, 2, 2, 2, 0, 2],
        )

    def test_unresolved_rows_are_clustered_and_new_subtypes_map_to_zero(self):
        model, data, validation_results = calibration_fixture()
        data.iloc[7] = data.iloc[6]
        figure = plt.figure()

        with patch.object(
            dgcytof_cli.DGCyTOF,
            'dimensionality_reduction_and_clustering',
            return_value=(
                np.asarray([[0.0, 0.0]]),
                np.asarray([3]),
                ['New Subtype 5'],
                figure,
            ),
        ) as clustering:
            result = dgcytof_cli.classify_sample(
                model,
                data,
                [1, 2],
                validation_results,
            )

        np.testing.assert_array_equal(
            clustering.call_args.args[0],
            data.iloc[[6]].to_numpy(dtype=np.float32),
        )
        self.assertEqual(result['predictions'].tolist()[-2:], [0, 2])
        self.assertEqual(result['clustering']['cluster_counts'], {'3': 1})
        self.assertEqual(result['rejection_count'], 1)
        self.assertNotIn(figure.number, plt.get_fignums())

    def test_clustering_records_when_no_cells_remain_unresolved(self):
        calibration = {
            'predictions': np.asarray([1, 2]),
            'unresolved_positions': [],
            'unresolved_index': [],
            'threshold': 0.5,
            'confident_count': 2,
            'initial_uncertain_count': 0,
            'promoted_count': 0,
            'unresolved_count': 0,
        }
        with patch.object(
            dgcytof_cli,
            'calibrate_with_indices',
            return_value=calibration,
        ), patch.object(
            dgcytof_cli.DGCyTOF,
            'dimensionality_reduction_and_clustering',
        ) as clustering:
            result = dgcytof_cli.classify_sample(
                object(), pd.DataFrame([[1.0], [2.0]]), [1, 2], []
            )

        clustering.assert_not_called()
        self.assertEqual(
            result['clustering']['reason'],
            'no cells remained unresolved after calibration',
        )

    def test_cli_fails_closed_when_vendored_import_is_hidden(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            isolated_runner = Path(tmpdir) / 'dgcytof_cli.py'
            output_dir = Path(tmpdir) / 'output'
            output_dir.mkdir()
            shutil.copy2(MODULE_PATH, isolated_runner)

            completed = subprocess.run(
                [sys.executable, str(isolated_runner), '--help'],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                'genuine vendored DGCyTOF package unavailable',
                completed.stderr,
            )
            self.assertEqual(list(output_dir.glob('*_predicted_labels.tar.gz')), [])

    def test_pending_archive_is_created_on_the_output_filesystem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / 'prediction.csv'
            output = root / 'output' / 'predictions.tar.gz'
            output.parent.mkdir()
            source.write_text('1\n', encoding='utf-8')

            with patch.object(dgcytof_cli.os, 'replace') as replace:
                dgcytof_cli.write_prediction_archive([str(source)], str(output))

            pending, destination = map(Path, replace.call_args.args)
            self.assertEqual(pending.parent, destination.parent)
            self.assertEqual(destination, output)

    def test_missing_pytorch_fails_instead_of_using_another_model(self):
        train = pd.DataFrame({'marker': [0.0, 1.0, 2.0, 3.0]})
        labels = pd.Series([1, 1, 2, 2])

        with patch.object(dgcytof_cli, 'TORCH_AVAILABLE', False):
            with self.assertRaisesRegex(RuntimeError, 'dependencies are unavailable'):
                dgcytof_cli.train_dgcytof(train, labels)


if __name__ == '__main__':
    unittest.main()

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / 'dgcytof_cli.py'
SPEC = importlib.util.spec_from_file_location('dgcytof_cli', MODULE_PATH)
dgcytof_cli = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(dgcytof_cli)


class DependencyContractTests(unittest.TestCase):
    def test_missing_pytorch_fails_instead_of_using_another_model(self):
        train = pd.DataFrame({'marker': [0.0, 1.0, 2.0, 3.0]})
        labels = pd.Series([1, 1, 2, 2])

        with patch.object(dgcytof_cli, 'TORCH_AVAILABLE', False):
            with self.assertRaisesRegex(RuntimeError, 'dependencies are unavailable'):
                dgcytof_cli.train_dgcytof(train, labels)


if __name__ == '__main__':
    unittest.main()

"""Check Candida growth adaptation, source masks, and held-out validation."""

import ast
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from generate_candida_intermedia_data import (
    BASAL_EXCHANGES, CARBON_EXCHANGES, INPUT_EXCHANGES, OXYGEN_EXCHANGE,
)


ROOT = Path(__file__).resolve().parents[1]


class TinyReservoir(nn.Module):
    """Cheap differentiable biomass output with the real Candida input order."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.03))
        self.register_buffer('input_token_indices', torch.arange(len(INPUT_EXCHANGES)))

    def forward(self, tokens, output_subset=None):
        self.last_tokens = tokens.detach().clone()
        biomass = -self.weight * tokens[:, self.input_token_indices[8:], 0].sum(dim=1)
        return biomass[:, None, None].expand(-1, tokens.shape[1], 1), torch.arange(tokens.shape[1])


class CandidaGrowthNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / 'candida_intermedia_model_testing.ipynb').read_text(encoding='utf-8'))

    def setUp(self):
        self.ns = dict(np=np, pd=pd, torch=torch, nn=nn, deepcopy=deepcopy, re=re, json=json,
            StandardScaler=StandardScaler, Ridge=Ridge, make_pipeline=make_pipeline,
            r2_score=r2_score, mean_squared_error=mean_squared_error,
            mean_absolute_error=mean_absolute_error, train_test_split=train_test_split,
            BASAL_EXCHANGES=BASAL_EXCHANGES, CARBON_EXCHANGES=CARBON_EXCHANGES,
            INPUT_EXCHANGES=INPUT_EXCHANGES, OXYGEN_EXCHANGE=OXYGEN_EXCHANGE,
            outputs=[f'{rid}_flux' for rid in INPUT_EXCHANGES] + ['BIOMASS_C_flux'],
            input_cols=list(INPUT_EXCHANGES), device=torch.device('cpu'), model=TinyReservoir(),
            GROWTH_SETTINGS=dict(hidden=(8, 4), dropout=0.1, learning_rate=0.01,
                weight_decay=0.001, epochs=2, patience=2, min_delta=1e-7,
                batch_size=2, inner_fraction=0.25, seed=42))
        for cell in self.notebook['cells'][14:]:
            if cell['cell_type'] == 'code':
                nodes = [node for node in ast.parse(''.join(cell['source'])).body
                         if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
                exec(compile(ast.Module(body=nodes, type_ignores=[]), '<growth-notebook>', 'exec'), self.ns)

    def test_all_sandra_targets_are_retained(self):
        path = ROOT / 'metabolic_models/Growth_data_for_ML_Helsinki_Uni.csv'
        if not path.is_file():
            self.skipTest('Private experimental CSV is unavailable.')
        self.ns.update(GROWTH_CSV=path, GROWTH_TARGET='growth_max_sandra', display=lambda frame: None)
        cell = next(c for c in self.notebook['cells'] if c['id'] == 'ci-growth-data')
        with contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(cell['source']), self.ns)
        data, y = self.ns['growth'], self.ns['growth_y']
        self.assertEqual(len(data), 29)
        self.assertEqual(data['original_fit_failed'].sum(), 2)
        np.testing.assert_array_equal(y, data['growth_max_sandra'].to_numpy(np.float32))
        self.assertTrue(np.isfinite(y).all())
        self.assertFalse(set(self.ns['response_columns']) & set(self.ns['chemical_features']))
        groups = data['medium_id'].map(self.ns['growth_medium_group'])
        self.assertEqual(groups.nunique(), 22)
        for train, test in LeaveOneGroupOut().split(data, groups=groups):
            self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[test]))

    def test_signed_bounds_masks_and_frozen_reservoir(self):
        reservoir = self.ns['model']
        initial = reservoir.weight.detach().clone()
        network = self.ns['CandidaGrowthMLP'](25, reservoir).train()
        presence = torch.ones(2, 9)
        presence[:, 0] = 0
        predicted, rates = network(torch.ones(2, 25), presence)
        tokens = reservoir.last_tokens[:, :, 0]
        self.assertFalse(reservoir.training)
        self.assertFalse(reservoir.weight.requires_grad)
        self.assertTrue((tokens[:, :8] == -1000).all())
        self.assertTrue((rates[:, 0] == 0).all())
        self.assertTrue((rates[:, 1:] > 0).all())
        for j, rid in enumerate([*CARBON_EXCHANGES.values(), OXYGEN_EXCHANGE]):
            index = self.ns['outputs'].index(f'{rid}_flux')
            torch.testing.assert_close(tokens[:, index], -rates[:, j])
        predicted.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.front.parameters()))
        self.assertIsNone(reservoir.weight.grad)
        torch.testing.assert_close(reservoir.weight, initial)

    def test_scaling_and_training_ignore_unused_test_target(self):
        x = np.arange(18, dtype=np.float32).reshape(6, 3)
        x[-1] = 1000
        y = np.linspace(0.02, 0.2, 6, dtype=np.float32)
        presence = np.ones((6, 9), dtype=np.float32)
        fit = self.ns['fit_growth_mlp']
        first, scaler, first_epoch, _ = fit(x, y, presence, [0, 1, 2, 3], [4])
        np.testing.assert_allclose(scaler.mean_, x[:4].mean(axis=0))
        changed = y.copy()
        changed[5] = 9999
        second, _, second_epoch, _ = fit(x, changed, presence, [0, 1, 2, 3], [4])
        self.assertEqual(first_epoch, second_epoch)
        for key, value in first.front.state_dict().items():
            torch.testing.assert_close(value, second.front.state_dict()[key])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            fit(x, y, presence, [0, 1], [1, 2])

    def test_each_outer_target_is_predicted_once_by_every_method(self):
        random = np.random.default_rng(9)
        features = list(CARBON_EXCHANGES) + [f'chemical_{i}' for i in range(17)]
        data = pd.DataFrame(random.uniform(0, 5, (6, 25)), columns=features)
        data['medium_id'] = [f'medium-{i}' for i in range(6)]
        data['original_fit_failed'] = [False, True, False, False, True, False]
        self.ns.update(growth=data, growth_y=np.linspace(0.01, 0.2, 6, dtype=np.float32),
            chemical_features=features, carbon_features=list(CARBON_EXCHANGES),
            growth_presence=np.ones((6, 9), dtype=np.float32),
            CHECKPOINT_PATH=Path('checkpoint.pth'), GROWTH_CSV=Path('experiment.csv'),
            MODEL_PATH=Path('model.sbml'), GROWTH_TARGET='growth_max_sandra',
            file_sha256=lambda path: f'test-hash:{path.name}')
        original_fit = self.ns['fit_growth_mlp']
        calls = []

        def recorded_fit(*args, **kwargs):
            validation = args[4] if len(args) > 4 else kwargs.get('validation_ids', [])
            calls.append((set(args[3]), set(validation)))
            return original_fit(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            self.ns['growth_dir'] = Path(temporary)
            with patch.dict(self.ns, fit_growth_mlp=recorded_fit), contextlib.redirect_stdout(io.StringIO()):
                result = self.ns['run_growth_cv'](LeaveOneOut().split(data), 'test')
            provenance = json.loads((Path(temporary) / 'test/provenance.json').read_text())
            self.assertEqual(provenance['target'], 'growth_max_sandra')
            self.assertEqual(provenance['n_conditions'], 6)
            self.assertEqual(provenance['folds'], result['folds'])
            exported = pd.read_csv(Path(temporary) / 'test/predictions.csv')
            self.assertEqual(len(exported), 30)
        predictions = result['predictions']
        self.assertEqual(len(predictions), 30)
        self.assertTrue(predictions.groupby(['method', 'row']).size().eq(1).all())
        self.assertTrue(np.isfinite(predictions.prediction).all())
        self.assertEqual(len(result['contexts']), 12)
        for fold in result['folds']:
            held_out = set(fold['test'])
            for train, validation in calls[fold['fold'] * 6:(fold['fold'] + 1) * 6]:
                self.assertFalse(held_out & (train | validation))
            self.assertFalse(set(fold['inner_train']) & set(fold['inner_validation']))
        self.assertFalse(self.ns['model'].weight.requires_grad)


if __name__ == '__main__':
    unittest.main()

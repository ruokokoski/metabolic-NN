"""Numerical and data-contract checks for the Candida pilot generator."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cobra.flux_analysis import pfba
from cobra.io import read_sbml_model
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients
import numpy as np

import generate_candida_intermedia_data as sampler


class CandidaSamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = read_sbml_model(str(sampler.DEFAULT_MODEL))
        cls.outputs = sorted(r.id for r in cls.model.reactions)
        cls.stoichiometry = create_stoichiometric_matrix(cls.model)

    def test_cli_defaults_and_validation(self):
        train = sampler.parse_args([])
        test = sampler.parse_args(["--dataset", "test"])
        self.assertEqual((train.n_samples, train.seed), (100_000, 42))
        self.assertEqual((test.n_samples, test.seed), (50_000, 9))
        self.assertEqual((train.flux_solver_mode, train.pfba_fraction_of_optimum), ("pfba", 0.999))
        override = sampler.parse_args(["--dataset", "test", "--n-samples", "4", "--seed", "21"])
        self.assertEqual((override.n_samples, override.seed), (4, 21))
        for flags in (["--n-samples", "0"], ["--seed", "-1"],
                      ["--carbon-min", "0"], ["--oxygen-max", "nan"],
                      ["--carbon-min", "11"], ["--max-additional-carbons", "8"],
                      ["--pfba-fraction-of-optimum", "nan"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    sampler.parse_args(flags)

    def test_signed_sampling_cardinality_and_reproducibility(self):
        args = sampler.parse_args([])
        first = np.random.default_rng(42)
        repeated = np.random.default_rng(42)
        cardinalities = []
        for _ in range(800):
            medium = sampler.draw_medium(first, args)
            self.assertEqual(medium, sampler.draw_medium(repeated, args))
            self.assertEqual(tuple(medium), sampler.INPUT_EXCHANGES)
            for rid in sampler.BASAL_EXCHANGES:
                self.assertEqual(medium[rid], -1000.0)
            for rid in (sampler.GLUCOSE_EXCHANGE, sampler.OXYGEN_EXCHANGE):
                self.assertTrue(-10 <= medium[rid] <= -0.1)
            optional_values = [medium[rid] for rid in sampler.OPTIONAL_CARBON_EXCHANGES]
            self.assertTrue(all(value == 0 or -10 <= value <= -0.1 for value in optional_values))
            cardinalities.append(sum(value < 0 for value in optional_values))
        self.assertEqual(set(cardinalities), set(range(8)))
        self.assertNotIn("UF02971_E", sampler.INPUT_EXCHANGES)
        glucose_only = sampler.parse_args(["--max-additional-carbons", "0"])
        self.assertTrue(all(sampler.draw_medium(first, glucose_only)[rid] == 0
                            for rid in sampler.OPTIONAL_CARBON_EXCHANGES))
        log_args = sampler.parse_args(["--uptake-distribution", "loguniform"])
        self.assertLess(sampler.draw_medium(first, log_args)[sampler.GLUCOSE_EXCHANGE], 0)

    def test_real_solver_feasibility_pfba_and_bound_preservation(self):
        model = self.model
        sampler.validate_model(model)
        original = {r.id: r.bounds for r in model.reactions}
        for mode in ("fba", "pfba"):
            args = sampler.parse_args(["--flux-solver-mode", mode])
            medium = sampler.draw_medium(np.random.default_rng(42), args)
            with model:
                model.medium = {}
                fluxes, status = sampler.solve_medium(model, medium, self.outputs, args)
                self.assertEqual(status, "optimal")
                self.assertIsNotNone(fluxes)
                # solve_medium restored its temporary bounds; rebuild the exact saved condition.
                for rid, lower in medium.items():
                    model.reactions.get_by_id(rid).lower_bound = lower
                objective = linear_reaction_coefficients(model)
                flux_by_id = dict(zip(self.outputs, fluxes))
                growth = sum(c * flux_by_id[r.id] for r, c in objective.items())
                optimum = model.slim_optimize(error_value=None)
                fraction = args.pfba_fraction_of_optimum if mode == "pfba" else 1.0
                self.assertGreaterEqual(growth, fraction * optimum - 1e-7)
                ordered = np.array([flux_by_id[r.id] for r in model.reactions])
                np.testing.assert_allclose(self.stoichiometry @ ordered, 0, atol=1e-7)
                for r in model.reactions:
                    self.assertGreaterEqual(flux_by_id[r.id], r.lower_bound - 1e-7)
                    self.assertLessEqual(flux_by_id[r.id], r.upper_bound + 1e-7)
                    self.assertEqual(r.upper_bound, original[r.id][1])
                for r in model.exchanges:
                    self.assertEqual(r.lower_bound, medium.get(r.id, 0.0))
                self.assertLess(flux_by_id[sampler.GLUCOSE_EXCHANGE], 0)
                if mode == "pfba":
                    reference = pfba(model, fraction_of_optimum=args.pfba_fraction_of_optimum)
                    self.assertAlmostEqual(float(np.abs(fluxes).sum()), reference.objective_value, places=6)
            self.assertEqual({r.id: r.bounds for r in model.reactions}, original)

    def test_written_schema_retries_provenance_and_independent_test_seed(self):
        solve = sampler.solve_medium
        calls = 0

        def reject_first(*args):
            nonlocal calls
            calls += 1
            return (None, "infeasible") if calls == 1 else solve(*args)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sampler, "read_sbml_model", return_value=self.model), \
                contextlib.redirect_stdout(io.StringIO()):
            flags = ["--n-samples", "3", "--output-dir", directory, "--progress-interval", "0"]
            with patch.object(sampler, "solve_medium", side_effect=reject_first):
                path = sampler.main(flags)
            with path.open() as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames[:17], list(sampler.INPUT_EXCHANGES))
                self.assertEqual(reader.fieldnames[17:], [f"{rid}_flux" for rid in self.outputs])
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(self.outputs), 1971)
            metadata = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual((metadata["accepted_samples"], metadata["attempts"]), (3, 4))
            self.assertEqual(metadata["rejected_by_status"], {"infeasible": 1})
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["output_reaction_ids"], self.outputs)
            self.assertEqual(metadata["input_token_indices"], [self.outputs.index(rid) for rid in sampler.INPUT_EXCHANGES])
            self.assertTrue(all(float(row[sampler.GLUCOSE_EXCHANGE]) < 0 for row in rows))
            self.assertFalse(list(Path(directory).glob('*_temp.csv')))
            test_path = sampler.main(flags + ["--dataset", "test"])
            with test_path.open() as handle:
                test_rows = list(csv.DictReader(handle))
            train_inputs = {tuple(row[rid] for rid in sampler.INPUT_EXCHANGES) for row in rows}
            test_inputs = {tuple(row[rid] for rid in sampler.INPUT_EXCHANGES) for row in test_rows}
            self.assertTrue(train_inputs.isdisjoint(test_inputs))
            self.assertEqual(json.loads(test_path.with_suffix('.json').read_text())["arguments"]["seed"], 9)
            with self.assertRaises(FileExistsError):
                sampler.main(flags)

    def test_attempt_limit_preserves_partial_provenance(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sampler, "read_sbml_model", return_value=self.model), \
                patch.object(sampler, "solve_medium", return_value=(None, "infeasible")), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                sampler.main(["--n-samples", "1", "--output-dir", directory,
                              "--max-attempt-multiplier", "2"])
            metadata_path, = Path(directory).glob('*.json')
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["status"], "incomplete")
            self.assertEqual(metadata["attempts"], 2)
            self.assertEqual(metadata["accepted_samples"], 0)
            self.assertTrue((Path(directory) / metadata["data_file"]).exists())


if __name__ == "__main__":
    unittest.main()

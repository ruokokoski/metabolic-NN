"""Integration checks for the deterministic AMN sweep and incomplete-grid handling."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import generate_AMN_sweep as sweep
import generate_ecoli_iML1515_AMN_data as reference


class AMNSweepTest(unittest.TestCase):
    def test_grid_and_medium_match_reference(self):
        model, bounds, outputs = reference.load_generation_model("models", sweep.parse_args([]).objective_reaction, 120)
        inputs = (reference.BASE_EXCHANGES + reference.FIXED_CARBON_EXCHANGES
                  + reference.AMINO_EXCHANGES + reference.CARBON_EXCHANGES)
        original_solve = sweep.solve_fluxes
        observed_statuses = []

        def checked_solve(model, mode, fraction):
            for exchange in model.exchanges:
                if exchange.id in reference.BASE_EXCHANGES:
                    expected = -10 if exchange.id != "EX_o2_e" else exchange.lower_bound
                elif exchange.id in reference.FIXED_CARBON_EXCHANGES + reference.AMINO_EXCHANGES:
                    expected = -2.2
                elif exchange.id == "EX_fru_e":
                    expected = exchange.lower_bound
                else:
                    expected = 0
                self.assertEqual(exchange.lower_bound, expected, exchange.id)
                self.assertEqual(exchange.upper_bound, max(0, bounds[exchange.id][1]))
            self.assertIn(model.reactions.EX_fru_e.lower_bound, -np.linspace(0.05, 2.2, 3))
            self.assertIn(model.reactions.EX_o2_e.lower_bound, -np.linspace(1, 10, 3))
            solution = original_solve(model, mode, fraction)
            observed_statuses.append(solution.status)
            return solution

        with tempfile.TemporaryDirectory() as folder, patch.object(sweep, "solve_fluxes", checked_solve):
            with contextlib.redirect_stdout(io.StringIO()):
                sweep.main(["--data-dir", folder, "--carbon-levels", "3", "--oxygen-levels", "3",
                            "--solver-reset-interval", "4"])
            data = pd.read_csv(Path(folder) / "iML1515_AMN_sweep_9_samples.csv")
            meta = pd.read_csv(Path(folder) / "iML1515_AMN_sweep_9_metadata.csv")
            self.assertEqual(data.shape, (9, 38 + 2712))
            self.assertEqual(list(data.columns), inputs + [f"{r}_flux" for r in outputs])
            self.assertTrue(np.isfinite(data.to_numpy()).all())
            np.testing.assert_array_equal(meta.sample_id, np.arange(9))
            np.testing.assert_array_equal(meta.sweep_fructose_index, np.repeat(np.arange(3), 3))
            np.testing.assert_array_equal(meta.sweep_oxygen_index, np.tile(np.arange(3), 3))
            np.testing.assert_allclose(data.EX_fru_e, np.repeat(np.linspace(0.05, 2.2, 3), 3))
            np.testing.assert_allclose(data.EX_o2_e, np.tile(np.linspace(1, 10, 3), 3))
            np.testing.assert_allclose(meta.sweep_fructose_rate, data.EX_fru_e)
            np.testing.assert_allclose(meta.sweep_oxygen_rate, data.EX_o2_e)
            for exchange in inputs:
                if exchange == "EX_fru_e" or exchange == "EX_o2_e":
                    continue
                expected = (10 if exchange in reference.BASE_EXCHANGES else
                            2.2 if exchange in reference.FIXED_CARBON_EXCHANGES + reference.AMINO_EXCHANGES else 0)
                np.testing.assert_array_equal(data[exchange], np.full(9, expected))
            with self.assertRaises(FileExistsError):
                sweep.main(["--data-dir", folder, "--carbon-levels", "3", "--oxygen-levels", "3"])
        self.assertEqual(observed_statuses, ["optimal"] * 9)

    def test_incomplete_grid_is_not_published(self):
        original = sweep.generate_sweep_sample

        def fail_first(model, bounds, outputs, args, carbon, oxygen):
            if carbon == args.carbon_min and oxygen == args.oxygen_min:
                return None, "infeasible"
            return original(model, bounds, outputs, args, carbon, oxygen)

        with tempfile.TemporaryDirectory() as folder, patch.object(sweep, "generate_sweep_sample", fail_first):
            log = io.StringIO()
            with contextlib.redirect_stdout(log), self.assertRaisesRegex(RuntimeError, "3/4 optimal"):
                sweep.main(["--data-dir", folder, "--carbon-levels", "2", "--oxygen-levels", "2"])
            self.assertIn("FAILED sample_id=0", log.getvalue())
            self.assertFalse((Path(folder) / "iML1515_AMN_sweep_4_samples.csv").exists())
            self.assertFalse((Path(folder) / "iML1515_AMN_sweep_4_metadata.csv").exists())
            meta = pd.read_csv(Path(folder) / "iML1515_AMN_sweep_4_metadata.partial.csv")
            np.testing.assert_array_equal(meta.sample_id, [1, 2, 3])

    def test_argument_validation(self):
        defaults = sweep.parse_args([])
        self.assertEqual(defaults.carbon_levels * defaults.oxygen_levels, 10000)
        self.assertEqual(defaults.flux_solver_mode, "fba")
        for argv in (["--carbon-levels", "1"], ["--oxygen-levels", "1"],
                     ["--carbon-min", "2.2"], ["--oxygen-min", "10"],
                     ["--oxygen-max", "nan"], ["--carbon-min", "-1"],
                     ["--pfba-fraction-of-optimum", "0"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                sweep.validate_args(sweep.parse_args(argv))


if __name__ == "__main__":
    unittest.main()

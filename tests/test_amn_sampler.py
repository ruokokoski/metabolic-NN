"""Checks for experimental-pattern allocation, uptake caps, and pFBA output."""

import contextlib
import csv
import io
from itertools import islice
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import generate_ecoli_iML1515_AMN_data as sampler


class AMNSamplerTest(unittest.TestCase):
    def test_experimental_patterns_and_balanced_counts(self):
        args = sampler.parse_args([])
        self.assertEqual(args.n_samples, 11000)
        self.assertEqual(args.flux_solver_mode, "pfba")
        conditions = sampler.load_experimental_conditions(args.conditions_csv)
        with open(args.conditions_csv, newline="") as fh:
            expected = {
                tuple(ex for ex in sampler.CARBON_EXCHANGES if float(row[ex + "_i"]) == 1)
                for row in csv.DictReader(fh)
            }
        self.assertEqual(set(conditions), expected)
        for count in (7, 113, 11000):
            draws = list(islice(sampler.iter_experimental_conditions(
                np.random.default_rng(9), conditions), count))
            counts = [draws.count(condition) for condition in conditions]
            self.assertEqual(max(counts) - min(counts), int(count % 110 != 0))
            self.assertEqual(sum(counts), count)
        repeated = list(islice(sampler.iter_experimental_conditions(
            np.random.default_rng(9), conditions), 7))
        self.assertEqual(repeated, draws[:7])

    def test_loguniform_rates_and_invalid_input(self):
        rng = np.random.default_rng(9)
        for lower, upper in ((0.05, 10), (1, 25)):
            actual = [sampler.random_rate(rng, lower, upper, log_uniform=True) for _ in range(20000)]
            self.assertTrue(all(lower <= value <= upper for value in actual))
            logs = np.log(actual)
            self.assertAlmostEqual(float(logs.mean()), (np.log(lower) + np.log(upper)) / 2, delta=0.04)
            self.assertAlmostEqual(float(logs.var()), np.log(upper / lower) ** 2 / 12, delta=0.08)
        for argv in (["--carbon-rate-min", "0"], ["--oxygen-rate-max", "nan"],
                     ["--carbon-rate-max", "0.01"], ["--n-samples", "0"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                sampler.validate_args(sampler.parse_args(argv))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            original = Path(sampler.parse_args([]).conditions_csv).read_text()
            lines = original.splitlines()
            path.write_text("\n".join(lines[:-1] + [lines[1]]) + "\n")
            with self.assertRaisesRegex(ValueError, "110 unique"):
                sampler.load_experimental_conditions(path)
            path.write_text(original.replace("EX_rib__D_e_i", "missing_ribose"))
            with self.assertRaisesRegex(ValueError, "column for EX_rib"):
                sampler.load_experimental_conditions(path)

    def test_failed_draw_retries_same_condition(self):
        attempts = []

        def generate(**kwargs):
            condition = kwargs["carbon_subset"]
            attempts.append(condition)
            if len(attempts) == 1:
                return None, "infeasible"
            return {ex: 1.0 for ex in condition}, "optimal"

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(sampler, "load_generation_model", return_value=(
                    SimpleNamespace(objective="test"), {}, [])), \
                patch.object(sampler, "validate_setup"), \
                patch.object(sampler, "generate_training_sample", side_effect=generate), \
                contextlib.redirect_stdout(io.StringIO()):
            sampler.main(["--data-dir", folder, "--n-samples", "113"])
            data = pd.read_csv(Path(folder) / "iML1515_AMN_training_data_113_samples.csv")
        self.assertEqual(attempts[0], attempts[1])
        patterns = [tuple(ex for ex in sampler.CARBON_EXCHANGES if row[ex] > 0)
                    for _, row in data.iterrows()]
        self.assertEqual(len(set(patterns)), 110)
        self.assertEqual(sorted(patterns.count(c) for c in set(patterns)), [1] * 107 + [2] * 3)

    def test_actual_pfba_medium_and_csv(self):
        original_solve = sampler.solve_fluxes
        checked = []
        output_columns = []

        def solve(model, flux_solver_mode, pfba_fraction_of_optimum):
            self.assertEqual(flux_solver_mode, "pfba")
            self.assertEqual(pfba_fraction_of_optimum, 0.999)
            for exchange in model.exchanges:
                uptake = -exchange.lower_bound
                if exchange.id == "EX_o2_e":
                    self.assertTrue(1 <= uptake <= 25)
                elif exchange.id in sampler.BASE_EXCHANGES:
                    self.assertEqual(uptake, 10)
                elif exchange.id in sampler.FIXED_CARBON_EXCHANGES + sampler.AMINO_EXCHANGES:
                    self.assertEqual(uptake, 2.2)
                elif exchange.id in sampler.CARBON_EXCHANGES:
                    self.assertTrue(uptake == 0 or 0.05 <= uptake <= 10)
                else:
                    self.assertEqual(uptake, 0, exchange.id)
            result = original_solve(model, flux_solver_mode, pfba_fraction_of_optimum)
            checked.append(result.status)
            output_columns[:] = [reaction.id + "_flux" for reaction in model.reactions]
            return result

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(sampler, "solve_fluxes", side_effect=solve), \
                contextlib.redirect_stdout(io.StringIO()):
            sampler.main(["--data-dir", folder, "--n-samples", "2"])
            data = pd.read_csv(Path(folder) / "iML1515_AMN_training_data_2_samples.csv")
            self.assertEqual(data.shape, (2, 38 + 2712))
            self.assertEqual(list(data.columns), sampler.BASE_EXCHANGES
                             + sampler.FIXED_CARBON_EXCHANGES + sampler.AMINO_EXCHANGES
                             + sampler.CARBON_EXCHANGES + output_columns)
            self.assertTrue(np.isfinite(data.to_numpy()).all())
            conditions = sampler.load_experimental_conditions(sampler.parse_args([]).conditions_csv)
            for _, row in data.iterrows():
                condition = tuple(ex for ex in sampler.CARBON_EXCHANGES if row[ex] > 0)
                self.assertIn(condition, conditions)
            self.assertTrue((data.BIOMASS_Ec_iML1515_core_75p37M_flux > 0).all())
            with self.assertRaises(FileExistsError):
                sampler.main(["--data-dir", folder, "--n-samples", "2"])
        self.assertEqual(checked, ["optimal", "optimal"])


if __name__ == "__main__":
    unittest.main()

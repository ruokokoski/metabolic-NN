"""Check core input bounds against the solved metabolic condition."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from cobra.io import read_sbml_model
from cobra.flux_analysis import pfba
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients

import generate_ecoli_core_data as sampler


ROOT = Path(__file__).resolve().parents[1]


class CoreSamplerTest(unittest.TestCase):
    def test_cli_defaults_and_overrides(self):
        args = sampler.parse_args([])
        self.assertEqual((args.n_samples, args.seed, args.flux_solver_mode),
                         (10_000_000, 42, "pfba"))
        self.assertEqual(args.pfba_fraction_of_optimum, 0.999)
        args = sampler.parse_args(["--n-samples", "3", "--seed", "9",
                                   "--flux-solver-mode", "fba"])
        self.assertEqual((args.n_samples, args.seed, args.flux_solver_mode), (3, 9, "fba"))
        for flags in (["--n-samples", "0"], ["--pfba-fraction-of-optimum", "0"],
                      ["--pfba-fraction-of-optimum", "1.1"],
                      ["--pfba-fraction-of-optimum", "nan"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    sampler.parse_args(flags)

    def test_signed_inputs_and_unchanged_solver_targets(self):
        model = read_sbml_model(str(ROOT / "models/e_coli_core.xml"))
        outputs = [rxn.id for rxn in model.reactions]
        original_bounds = {rxn.id: rxn.bounds for rxn in model.exchanges}
        optimize = model.optimize
        observed = {}

        def solve():
            observed["bounds"] = {rxn.id: rxn.bounds for rxn in model.exchanges}
            observed["solution"] = optimize()
            return observed["solution"]

        with patch.object(sampler, "model", model, create=True), \
                patch.object(sampler, "random_rate", side_effect=[10.0, 10.0, 20.0, 50.0]), \
                patch.object(model, "optimize", side_effect=solve):
            sample = sampler.generate_training_sample(
                ["EX_glc__D_e"], ["EX_nh4_e"], outputs, default_rate=50,
                flux_solver_mode="fba",
            )

        self.assertIsNotNone(sample)
        self.assertEqual(observed["solution"].status, "optimal")
        for ex, (lower, upper) in observed["bounds"].items():
            self.assertEqual(sample.get(ex, 0.0), lower)
            self.assertEqual(upper, original_bounds[ex][1])
        self.assertEqual(sample["EX_glc__D_e"], -10.0)
        self.assertEqual(sample["EX_o2_e"], -20.0)
        self.assertEqual(sample["EX_pi_e"], -50.0)
        self.assertLess(sample["EX_glc__D_e_flux"], 0.0)
        self.assertGreater(sample["EX_co2_e_flux"], 0.0)

        fluxes = np.array([sample[f"{rxn}_flux"] for rxn in outputs])
        np.testing.assert_array_equal(fluxes, observed["solution"].fluxes.loc[outputs])
        np.testing.assert_allclose(create_stoichiometric_matrix(model) @ fluxes, 0, atol=1e-7)
        self.assertEqual({rxn.id: rxn.bounds for rxn in model.exchanges}, original_bounds)

    def test_pfba_growth_fraction_and_flux_minimization(self):
        model = read_sbml_model(str(ROOT / "models/e_coli_core.xml"))
        outputs = [rxn.id for rxn in model.reactions]
        for fraction in (0.999, 1.0):
            with self.subTest(fraction=fraction), \
                    patch.object(sampler, "model", model, create=True), \
                    patch.object(sampler, "random_rate", side_effect=[10.0, 10.0, 20.0, 50.0]):
                sample = sampler.generate_training_sample(
                    ["EX_glc__D_e"], ["EX_nh4_e"], outputs, 50,
                    pfba_fraction_of_optimum=fraction,
                )
                self.assertIsNotNone(sample)
                with model:
                    for reaction in model.exchanges:
                        reaction.lower_bound = sample.get(reaction.id, 0.0)
                    optimum = model.slim_optimize(error_value=None)
                    reference = pfba(model, fraction_of_optimum=fraction)
                    fluxes = np.array([sample[f"{rxn}_flux"] for rxn in outputs])
                    growth = sum(coefficient * sample[f"{reaction.id}_flux"]
                                 for reaction, coefficient in linear_reaction_coefficients(model).items())
                    self.assertGreaterEqual(growth, fraction * optimum - 1e-7)
                    self.assertAlmostEqual(np.abs(fluxes).sum(), reference.objective_value,
                                           places=6)
                    np.testing.assert_allclose(create_stoichiometric_matrix(model) @ fluxes,
                                               0, atol=1e-7)
                    for reaction, flux in zip(model.reactions, fluxes):
                        self.assertGreaterEqual(flux, reaction.lower_bound - 1e-7)
                        self.assertLessEqual(flux, reaction.upper_bound + 1e-7)

    def test_generation_retries_and_is_reproducible(self):
        model = read_sbml_model(str(ROOT / "models/e_coli_core.xml"))
        generate = sampler.generate_training_sample
        calls = 0

        def reject_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            return None if calls == 1 else generate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
                contextlib.chdir(directory), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sampler, "load_model", return_value=model), \
                patch.object(sampler, "generate_training_sample", side_effect=reject_first):
            flags = ["--n-samples", "3", "--output-prefix", "core_test"]
            sampler.main(flags)
            path, = Path("data").glob("*_core_test_3_samples.csv")
            self.assertGreater(calls, 3)
            with path.open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(float(row["EX_o2_e"]) < 0 for row in rows))
            calls = 0
            sampler.main(flags)
            with path.open() as fh:
                repeated = list(csv.DictReader(fh))
            input_names = [name for name in rows[0] if not name.endswith("_flux")]
            output_names = [name for name in rows[0] if name.endswith("_flux")]
            for names, exact in ((input_names, True), (output_names, False)):
                first = [[float(row[name]) for name in names] for row in rows]
                second = [[float(row[name]) for name in names] for row in repeated]
                if exact:
                    np.testing.assert_array_equal(first, second)
                else:
                    np.testing.assert_allclose(first, second, rtol=1e-7, atol=1e-7)
            self.assertFalse(list(Path("data").glob("*temp.csv")))


if __name__ == "__main__":
    unittest.main()

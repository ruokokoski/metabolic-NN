"""Check the uncurated F pool, medium contract, and reproducible pFBA output."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from cobra.util.array import create_stoichiometric_matrix

import generate_ecoli_iML1515_E_data as reference
import generate_ecoli_iML1515_F_data as sampler


ROOT = Path(__file__).resolve().parents[1]


class FSamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = sampler.parse_args([])
        cls.model, cls.defaults, cls.outputs, _ = sampler.load_generation_model(
            str(ROOT / "models"), cls.args.objective_reaction, cls.args.solver_timeout_seconds
        )
        cls.inputs = sampler.build_input_columns(cls.model)
        cls.pool = sampler.build_selectable_exchanges(cls.model)

    def test_training_test_defaults_and_e_rates(self):
        args, e_args = self.args, reference.parse_args([])
        self.assertEqual((args.n_samples, args.seed, args.output_prefix),
                         (1_000_000, 42, "iML1515_F_training_data"))
        test_args = sampler.parse_args(["--dataset", "test"])
        self.assertEqual((test_args.n_samples, test_args.seed, test_args.output_prefix),
                         (50_000, 9, "iML1515_F_test_data"))
        self.assertEqual(args.max_sources, 10)
        for name, e_name in (("base_rate", "g_base_rate"),
                             ("source_rate_min", "g_organic_rate_min"),
                             ("source_rate_max", "g_organic_rate_max"),
                             ("oxygen_rate_min", "g_oxygen_rate_min"),
                             ("oxygen_rate_max", "g_oxygen_rate_max"),
                             ("pfba_fraction_of_optimum", "pfba_fraction_of_optimum")):
            self.assertEqual(getattr(args, name), getattr(e_args, e_name))
        for argv in (["--source-rate-min", "0"], ["--source-rate-min", "0.001"],
                     ["--oxygen-rate-max", "nan"], ["--source-rate-max", "inf"],
                     ["--source-rate-min", "11"], ["--max-sources", "0"],
                     ["--n-samples", "0"], ["--seed", "-1"],
                     ["--base-rate", "nan"], ["--pfba-fraction-of-optimum", "0"],
                     ["--max-attempt-multiplier", "nan"], ["--output-prefix", "../bad"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                sampler.validate_args(sampler.parse_args(argv))

    def test_pool_is_every_remaining_exchange_and_schema_is_ordered(self):
        excluded = set(reference.FIXED_BASE_EXCHANGES) | {"EX_o2_e"}
        all_exchanges = [reaction.id for reaction in self.model.exchanges]
        self.assertEqual(self.pool, [exchange for exchange in all_exchanges if exchange not in excluded])
        self.assertEqual((len(self.inputs), len(self.pool), len(self.outputs)), (331, 308, 2712))
        self.assertEqual(set(self.inputs), set(all_exchanges))
        self.assertEqual(len(self.inputs), len(set(self.inputs)))
        self.assertEqual(self.inputs[:5], reference.MINN_CONTEXT_EXCHANGES)
        self.assertTrue(set(reference.SELECTABLE_ORGANIC_EXCHANGES) <= set(self.pool))
        self.assertTrue({"EX_no3_e", "EX_cbl1_e", "EX_etoh_e"} <= set(self.pool))
        output_columns = [reaction + "_flux" for reaction in self.outputs]
        sampler.validate_generation_model(self.model, self.inputs, output_columns, self.args)
        with self.assertRaises(ValueError):
            sampler.validate_generation_model(self.model, self.inputs, output_columns,
                                               sampler.parse_args(["--max-sources", "309"]))

    def test_proposal_distribution_and_reproducibility(self):
        rng = np.random.default_rng(9)
        counts, rates, oxygen = [], [], []
        source_counts = dict.fromkeys(self.pool, 0)
        for _ in range(10_000):
            medium = sampler.draw_medium(rng, self.inputs, self.pool, self.args)
            selected = [exchange for exchange in self.pool if medium[exchange] > 0]
            counts.append(len(selected))
            oxygen.append(medium["EX_o2_e"])
            for exchange in selected:
                rates.append(medium[exchange])
                source_counts[exchange] += 1
        np.testing.assert_allclose(np.bincount(counts)[1:] / len(counts), 0.1, atol=0.01, rtol=0)
        expected_frequency = len(rates) / len(self.pool)
        np.testing.assert_allclose(list(source_counts.values()), expected_frequency,
                                   atol=5 * np.sqrt(expected_frequency), rtol=0)
        for values, lower, upper in ((rates, 0.05, 10), (oxygen, 1, 25)):
            self.assertTrue(all(lower <= value <= upper for value in values))
            self.assertTrue(all(round(value, 2) == value for value in values))
            self.assertAlmostEqual(float(np.log(values).mean()),
                                   (np.log(lower) + np.log(upper)) / 2, delta=0.04)
        first_rng, second_rng = np.random.default_rng(42), np.random.default_rng(42)
        for _ in range(20):
            self.assertEqual(sampler.draw_medium(first_rng, self.inputs, self.pool, self.args),
                             sampler.draw_medium(second_rng, self.inputs, self.pool, self.args))

    def test_medium_bounds_and_reset_include_inorganic_and_cobalamin(self):
        # These candidates must not be filtered out or converted to secretion caps.
        result = SimpleNamespace(status="optimal", fluxes=pd.Series(0.0, index=self.outputs))
        for source in ("EX_no3_e", "EX_cbl1_e", "EX_etoh_e", "EX_glc__D_e"):
            with patch.object(sampler, "draw_uniform_subset", return_value=[source]), \
                    patch.object(sampler, "pfba", return_value=result):
                sample, status = sampler.generate_training_sample(
                    self.model, np.random.default_rng(9), self.inputs, self.outputs,
                    self.defaults, self.pool, self.args
                )
            self.assertEqual(status, "accepted")
            self.assertEqual([exchange for exchange in self.pool if sample[exchange] > 0], [source])
            for reaction in self.model.exchanges:
                self.assertEqual(reaction.lower_bound, -sample[reaction.id])
                self.assertEqual(reaction.upper_bound, max(0, self.defaults[reaction.id][1]))
            for exchange in sampler.FIXED_BASE_EXCHANGES:
                self.assertEqual(sample[exchange], 10)
            self.assertTrue(1 <= sample["EX_o2_e"] <= 25)
            self.assertEqual(sample[self.args.objective_reaction + "_flux"], 0)
        result.fluxes.iloc[0] = np.nan
        with patch.object(sampler, "pfba", return_value=result):
            sample, status = sampler.generate_training_sample(
                self.model, np.random.default_rng(9), self.inputs, self.outputs,
                self.defaults, self.pool, self.args
            )
        self.assertIsNone(sample)
        self.assertEqual(status, "nonfinite_flux")

    def test_actual_pfba_training_test_csv_and_reproducibility(self):
        original_pfba = sampler.pfba
        checked = []
        stoichiometry = create_stoichiometric_matrix(self.model, array_type="lil")

        def solve(model, fraction_of_optimum):
            self.assertEqual(fraction_of_optimum, 0.999)
            optimum = model.slim_optimize()
            result = original_pfba(model, fraction_of_optimum=fraction_of_optimum)
            fluxes = result.fluxes.loc[self.outputs].to_numpy()
            self.assertLess(float(np.max(np.abs(stoichiometry @ fluxes))), 1e-5)
            lower = np.array([reaction.lower_bound for reaction in model.reactions])
            upper = np.array([reaction.upper_bound for reaction in model.reactions])
            self.assertTrue(np.all(fluxes >= lower - 1e-5))
            self.assertTrue(np.all(fluxes <= upper + 1e-5))
            self.assertGreaterEqual(result.fluxes[self.args.objective_reaction], optimum * 0.999 - 1e-5)
            checked.append(result.status)
            return result

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(sampler, "pfba", side_effect=solve), \
                contextlib.redirect_stdout(io.StringIO()):
            contents = []
            for dataset in ("training", "test"):
                argv = ["--dataset", dataset, "--data-dir", folder,
                        "--model-dir", str(ROOT / "models"), "--n-samples", "8"]
                sampler.main(argv)
                path = Path(folder) / f"iML1515_F_{dataset}_data_8_samples.csv"
                original = path.read_bytes()
                with path.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(reader.fieldnames,
                                     self.inputs + [reaction + "_flux" for reaction in self.outputs])
                    rows = list(reader)
                self.assertEqual(len(rows), 8)
                for row in rows:
                    self.assertTrue(np.isfinite([float(value) for value in row.values()]).all())
                    self.assertTrue(1 <= sum(float(row[exchange]) > 0 for exchange in self.pool) <= 10)
                    self.assertEqual(float(row["EX_co2_e"]), 10)
                    self.assertTrue(1 <= float(row["EX_o2_e"]) <= 25)
                with self.assertRaises(FileExistsError):
                    sampler.main(argv)
                sampler.main(argv + ["--overwrite-existing"])
                self.assertEqual(path.read_bytes(), original)
                contents.append(original)
            self.assertNotEqual(*contents)
            self.assertEqual(list(Path(folder).glob("*temp*")), [])
        self.assertEqual(checked, ["optimal"] * 32)

    def test_attempt_limit_preserves_partial_batch_and_reloads_after_failure(self):
        sample = dict.fromkeys(self.inputs + [reaction + "_flux" for reaction in self.outputs], 0.0)
        real_load = sampler.load_generation_model
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(sampler, "load_generation_model", wraps=real_load) as load, \
                patch.object(sampler, "generate_training_sample", side_effect=[
                    (sample, "accepted"), (None, "solver_status:infeasible"),
                    (None, "solver_status:infeasible")]), \
                contextlib.redirect_stdout(io.StringIO()):
            argv = ["--data-dir", folder, "--model-dir", str(ROOT / "models"),
                    "--n-samples", "3", "--max-attempt-multiplier", "1",
                    "--failure-reload-interval", "1"]
            with self.assertRaisesRegex(RuntimeError, "only 1/3 accepted samples"):
                sampler.main(argv)
            self.assertEqual(load.call_count, 2)
            self.assertFalse((Path(folder) / "iML1515_F_training_data_3_samples.csv").exists())
            paths = list(Path(folder).glob("*temp*"))
            self.assertEqual(len(paths), 1)
            with paths[0].open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)


if __name__ == "__main__":
    unittest.main()

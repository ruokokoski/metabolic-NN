"""Check the curated E distribution against the AMN medium contract."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from cobra.util.array import create_stoichiometric_matrix

import generate_ecoli_iML1515_AMN_data as amn
import generate_ecoli_iML1515_E_data as sampler


ROOT = Path(__file__).resolve().parents[1]


class ESamplerTest(unittest.TestCase):
    def test_pool_and_amn_defaults(self):
        with open(ROOT / "data/reference/iML1515_broad_organic_source_pool.csv", newline="") as fh:
            pool = [row["exchange_id"] for row in csv.DictReader(fh)]
        self.assertEqual(sampler.SELECTABLE_ORGANIC_EXCHANGES, pool)
        self.assertEqual(len(set(pool)), 48)
        self.assertTrue(set(amn.FIXED_CARBON_EXCHANGES + amn.AMINO_EXCHANGES) <= set(pool))
        self.assertEqual(sampler.FIXED_BASE_EXCHANGES,
                         [ex for ex in amn.BASE_EXCHANGES if ex != "EX_o2_e"])
        self.assertNotIn("EX_cbl1_e", sampler.build_input_columns())
        self.assertEqual(len(sampler.build_input_columns()), 72)
        args, reference = sampler.parse_args([]), amn.parse_args([])
        for actual, expected in (
            (args.g_base_rate, reference.default_rate),
            (args.g_organic_rate_min, reference.carbon_rate_min),
            (args.g_organic_rate_max, reference.carbon_rate_max),
            (args.g_oxygen_rate_min, reference.oxygen_rate_min),
            (args.g_oxygen_rate_max, reference.oxygen_rate_max),
            (args.pfba_fraction_of_optimum, reference.pfba_fraction_of_optimum),
        ):
            self.assertEqual(actual, expected)
        self.assertEqual(args.g_max_organic_sources, 10)
        self.assertEqual(args.output_prefix, "iML1515_E_training_data")
        self.assertEqual(args.n_samples, 1_000_000)
        self.assertEqual(args.seed, 42)

    def test_sampling_distributions_and_invalid_log_bounds(self):
        args = sampler.parse_args([])
        probabilities = sampler.build_g_k_probabilities(10, args.g_k_beta)
        np.testing.assert_allclose(probabilities[1:] / probabilities[:-1], 2 / 3)
        rng = np.random.default_rng(9)
        counts = [sampler.draw_g_source_count(rng, 10, probabilities) for _ in range(20000)]
        self.assertEqual(set(counts), set(range(1, 11)))
        np.testing.assert_allclose(np.bincount(counts)[1:] / len(counts), probabilities,
                                   atol=0.01, rtol=0)
        for lower, upper in ((0.05, 10), (1, 25)):
            rates = np.array([sampler.random_log_uniform_rate(rng, lower, upper)
                              for _ in range(20000)])
            self.assertTrue(np.all((lower <= rates) & (rates <= upper)))
            self.assertAlmostEqual(float(np.log(rates).mean()),
                                   (np.log(lower) + np.log(upper)) / 2, delta=0.04)
            self.assertAlmostEqual(float(np.log(rates).var()),
                                   np.log(upper / lower) ** 2 / 12, delta=0.08)
        for argv in (["--g-oxygen-rate-min", "0"], ["--g-organic-rate-max", "nan"],
                     ["--g-oxygen-rate-max", "inf"], ["--g-organic-rate-min", "0.001"],
                     ["--g-max-organic-sources", "49"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                sampler.validate_args(sampler.parse_args(argv), "e")

    def test_every_source_is_selectable_and_basal_bounds_match_amn(self):
        args = sampler.parse_args([])
        model, defaults, outputs, _ = sampler.load_generation_model(
            str(ROOT / "models"), args.objective_reaction, args.solver_timeout_seconds)
        sampler.validate_setup(model, sampler.build_input_columns(), [ex + "_flux" for ex in outputs])
        reference_args = amn.parse_args([])
        with patch.object(amn, "solve_fluxes", return_value=SimpleNamespace(status="optimal", fluxes={})):
            _, status = amn.generate_training_sample(model, np.random.default_rng(9), defaults,
                                                     outputs, reference_args, ("EX_fru_e",))
        self.assertEqual(status, "optimal")
        reference_bounds = {ex: model.reactions.get_by_id(ex).bounds
                            for ex in sampler.FIXED_BASE_EXCHANGES}
        probabilities = sampler.build_g_k_probabilities(10, args.g_k_beta)
        for source in sampler.SELECTABLE_ORGANIC_EXCHANGES:
            data = dict.fromkeys(sampler.build_input_columns(), 0.0)
            with patch.object(sampler, "draw_g_source_count", return_value=1), \
                    patch.object(sampler, "draw_uniform_subset", return_value=[source]):
                sampler.apply_g_regime(model, data, np.random.default_rng(9), defaults,
                                       probabilities, args)
            offered = [ex for ex in sampler.SELECTABLE_ORGANIC_EXCHANGES if data[ex] > 0]
            self.assertEqual(offered, [source])
            for ex in sampler.FIXED_BASE_EXCHANGES:
                self.assertEqual(model.reactions.get_by_id(ex).bounds, reference_bounds[ex])
            for reaction in model.exchanges:
                self.assertEqual(reaction.upper_bound, max(0, defaults[reaction.id][1]))
                if reaction.id == "EX_o2_e":
                    self.assertTrue(1 <= -reaction.lower_bound <= 25)
                elif reaction.id in sampler.FIXED_BASE_EXCHANGES:
                    self.assertEqual(reaction.lower_bound, -10)
                elif reaction.id == source:
                    self.assertTrue(0.05 <= -reaction.lower_bound <= 10)
                    self.assertEqual(data[source], -reaction.lower_bound)
                else:
                    self.assertEqual(reaction.lower_bound, 0, reaction.id)
        self.assertEqual(model.reactions.get_by_id("EX_cbl1_e").lower_bound, 0)

    def test_actual_pfba_csv_and_reproducibility(self):
        original_pfba = sampler.pfba
        checked = []
        output_columns = []

        def solve(model, fraction_of_optimum):
            self.assertEqual(fraction_of_optimum, 0.999)
            result = original_pfba(model, fraction_of_optimum=fraction_of_optimum)
            fluxes = result.fluxes.to_numpy()
            residual = create_stoichiometric_matrix(model) @ fluxes
            self.assertLess(float(np.max(np.abs(residual))), 1e-5)
            for reaction in model.reactions:
                self.assertGreaterEqual(result.fluxes[reaction.id], reaction.lower_bound - 1e-5)
                self.assertLessEqual(result.fluxes[reaction.id], reaction.upper_bound + 1e-5)
            output_columns[:] = [ex.id + "_flux" for ex in model.reactions]
            checked.append(result.status)
            return result

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(sampler, "pfba", side_effect=solve), \
                contextlib.redirect_stdout(io.StringIO()):
            argv = ["--data-dir", folder, "--model-dir", str(ROOT / "models"), "--n-samples", "8"]
            sampler.main(argv)
            path = Path(folder) / "iML1515_E_training_data_8_samples.csv"
            original = path.read_bytes()
            with path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                self.assertEqual(reader.fieldnames, sampler.build_input_columns() + output_columns)
                rows = list(reader)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(output_columns), 2712)
            for row in rows:
                self.assertTrue(np.isfinite([float(x) for x in row.values()]).all())
                count = sum(float(row[ex]) > 0 for ex in sampler.SELECTABLE_ORGANIC_EXCHANGES)
                self.assertTrue(1 <= count <= 10)
                self.assertEqual(float(row["EX_co2_e"]), 10)
                self.assertTrue(1 <= float(row["EX_o2_e"]) <= 25)
                self.assertEqual(float(row["EX_etoh_e"]), 0)
                self.assertNotIn("EX_cbl1_e", row)
            with self.assertRaises(FileExistsError):
                sampler.main(argv)
            sampler.main(argv + ["--overwrite-existing"])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).glob("*temp*")), [])
        self.assertEqual(checked, ["optimal"] * 16)


if __name__ == "__main__":
    unittest.main()

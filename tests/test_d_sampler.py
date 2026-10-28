"""D uses the current E pool/medium with current A/B task components."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from cobra.util.array import create_stoichiometric_matrix

import generate_ecoli_iML1515_A_data as a
import generate_ecoli_iML1515_B_data as b
import generate_ecoli_iML1515_D_data as d
import generate_ecoli_iML1515_E_data as e


ROOT = Path(__file__).resolve().parents[1]


class DSamplerTest(unittest.TestCase):
    def test_pool_schema_defaults_and_schedule(self):
        with (ROOT / "data/reference/iML1515_broad_organic_source_pool.csv").open(newline="") as fh:
            pool = [row["exchange_id"] for row in csv.DictReader(fh)]
        self.assertEqual(d.SELECTABLE_ORGANIC_EXCHANGES, pool)
        self.assertEqual(d.SELECTABLE_ORGANIC_EXCHANGES, e.SELECTABLE_ORGANIC_EXCHANGES)
        self.assertEqual(len(set(pool)), 48)
        self.assertEqual(d.build_input_columns(), e.build_input_columns())
        self.assertEqual(len(d.build_input_columns()), 72)
        self.assertNotIn("EX_cbl1_e", d.build_input_columns())
        args, reference = d.parse_args([]), e.parse_args([])
        for key, value in vars(reference).items():
            if key.startswith("g_") or key == "pfba_fraction_of_optimum":
                self.assertEqual(getattr(args, key), value, key)
        for key, expected in {
            "a_base_rate": 10, "a_carbon_rate_min": 0.05, "a_carbon_rate_max": 10,
            "a_oxygen_rate_min": 1, "a_oxygen_rate_max": 25,
            "a_fixed_glycerol_rate": 2.2, "a_amino_rate": 2.2,
            "a_max_carbon_sources": 4, "b_base_rate": 10,
        }.items():
            self.assertEqual(getattr(args, key), expected, key)
        self.assertEqual(d.build_b_cap_config(args), {
            "EX_glc__D_e": (1, 15), "EX_o2_e": (1, 25), "EX_co2_e": (0, 20),
            "EX_etoh_e": (0, 2), "EX_ac_e": (0, 8),
        })
        probabilities = d.build_g_k_probabilities(10, args.g_k_beta)
        np.testing.assert_allclose(probabilities[1:] / probabilities[:-1], 2 / 3)
        schedule = d.build_regime_schedule(1000, "d", args.task_fraction, np.random.default_rng(42))
        np.testing.assert_array_equal(np.bincount(schedule), [100, 100, 800])
        for argv in (["--g-oxygen-rate-min", "0"], ["--a-carbon-rate-min", "0"],
                     ["--a-oxygen-rate-min", "0"], ["--b-oxygen-cap-min", "0"],
                     ["--g-organic-rate-max", "nan"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                d.validate_args(d.parse_args(argv), "d")

    def test_all_48_sources_match_e_medium(self):
        args = d.parse_args([])
        model, defaults, outputs, _ = d.load_generation_model(
            str(ROOT / "models"), args.objective_reaction, 120)
        inputs = d.build_input_columns()
        d.validate_setup(model, inputs, [rxn + "_flux" for rxn in outputs])
        probabilities = d.build_g_k_probabilities(10, args.g_k_beta)
        for source in d.SELECTABLE_ORGANIC_EXCHANGES:
            snapshots = []
            for module in (d, e):
                data = dict.fromkeys(inputs, 0.0)
                with patch.object(module, "draw_g_source_count", return_value=1), \
                        patch.object(module, "draw_uniform_subset", return_value=[source]):
                    module.apply_g_regime(model, data, np.random.default_rng(9), defaults,
                                          probabilities, args)
                self.assertEqual([name for name in d.SELECTABLE_ORGANIC_EXCHANGES if data[name] > 0], [source])
                self.assertEqual(model.reactions.EX_co2_e.lower_bound, -10)
                self.assertEqual(model.reactions.EX_co2_e.upper_bound, defaults["EX_co2_e"][1])
                self.assertEqual(model.reactions.EX_cbl1_e.lower_bound, 0)
                snapshots.append((data, [reaction.bounds for reaction in model.exchanges]))
            self.assertEqual(snapshots[0], snapshots[1], source)

    def test_task_sampling_laws(self):
        args = d.parse_args([])
        model, defaults, _, _ = d.load_generation_model(str(ROOT / "models"), args.objective_reaction, 120)

        def record(model, data, exchange, rate):
            data[exchange] = rate

        rng = np.random.default_rng(9)
        a_counts, carbon_rates, a_oxygen, b_rows = [], [], [], []
        with patch.object(d, "reset_closed_medium"), patch.object(d, "restore_default_medium"), \
                patch.object(d, "set_uptake", side_effect=record), \
                patch.object(d, "set_secretion_cap", side_effect=record):
            for _ in range(6000):
                data = {}
                d.apply_a_regime(model, data, rng, defaults, args)
                selected = [name for name in a.CARBON_EXCHANGES if name in data]
                a_counts.append(len(selected))
                carbon_rates.extend(data[name] for name in selected)
                a_oxygen.append(data["EX_o2_e"])
                self.assertEqual(data["EX_glyc_e"], 2.2)
                self.assertTrue(all(data[name] == 2.2 for name in a.AMINO_EXCHANGES))
                data = {}
                d.apply_b_regime(model, data, rng, defaults, d.build_b_cap_config(args), args)
                b_rows.append([data[name] for name in b.SAMPLED_EXCHANGES])
        for count in range(1, 5):
            self.assertAlmostEqual(a_counts.count(count) / len(a_counts), 0.25, delta=0.025)
        for rates, low, high in ((carbon_rates, 0.05, 10), (a_oxygen, 1, 25)):
            values = np.asarray(rates)
            self.assertTrue(((values >= low) & (values <= high)).all())
            self.assertAlmostEqual(float(np.log(values).mean()), np.log(low * high) / 2, delta=0.06)
        caps = np.asarray(b_rows)
        for column, (low, high) in enumerate(d.build_b_cap_config(args).values()):
            values = caps[:, column]
            self.assertTrue(((values >= low) & (values <= high)).all())
            self.assertTrue(np.any(values != np.floor(values)))
            np.testing.assert_array_equal(values, np.round(values, 2))
            if column == 1:
                normalized = np.log(values / low) / np.log(high / low)
            else:
                normalized = (values - low) / (high - low)
            self.assertAlmostEqual(float(normalized.mean()), 0.5, delta=0.02)
            self.assertAlmostEqual(float(normalized.var()), 1 / 12, delta=0.008)

    def test_pfba_csv_quotas_survive_rejections(self):
        args = d.parse_args([])
        original = d.generate_training_sample
        failed, accepted = set(), []
        matrix = None

        def sample(**kwargs):
            nonlocal matrix
            regime = kwargs["regime"]
            if regime not in failed:
                failed.add(regime)
                return None, "solver_status:infeasible"
            data, status = original(**kwargs)
            if data is not None:
                accepted.append(regime)
                model = kwargs["model"]
                if matrix is None:
                    matrix = create_stoichiometric_matrix(model)
                fluxes = np.array([data[name + "_flux"] for name in kwargs["outputs"]])
                self.assertLess(float(np.abs(matrix @ fluxes).max()), 1e-5)
                for reaction, flux in zip(model.reactions, fluxes):
                    self.assertGreaterEqual(flux, reaction.lower_bound - 1e-5)
                    self.assertLessEqual(flux, reaction.upper_bound + 1e-5)
                self.assertAlmostEqual(data[args.objective_reaction + "_flux"],
                                       0.999 * model.slim_optimize(), delta=1e-6)
                self.assertEqual(model.reactions.EX_cbl1_e.lower_bound, 0)
            return data, status

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(d, "generate_training_sample", side_effect=sample), \
                contextlib.redirect_stdout(io.StringIO()):
            d.main(["--n-samples", "20", "--data-dir", folder,
                    "--model-dir", str(ROOT / "models")])
            path = Path(folder) / "iML1515_D_training_data_20_samples.csv"
            with path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                self.assertEqual(reader.fieldnames[:72], d.build_input_columns())
                rows = list(reader)
            self.assertEqual(len(rows), 20)
            self.assertEqual(len(reader.fieldnames), 72 + 2712)
            self.assertNotIn("EX_cbl1_e", reader.fieldnames)
            self.assertTrue(np.isfinite([[float(v) for v in row.values()] for row in rows]).all())
        self.assertEqual(failed, {"a", "b", "g"})
        self.assertEqual({name: accepted.count(name) for name in failed}, {"a": 2, "b": 2, "g": 16})

    def test_revised_evaluation_medium_and_old_checkpoint_rejection(self):
        import torch
        import generate_ecoli_iML1515_E_data_old as archived
        import iml1515_evaluation as ev

        args = d.parse_args([])
        _, _, outputs, _ = d.load_generation_model(str(ROOT / "models"), args.objective_reaction, 120)
        tokens = [name + "_flux" for name in outputs]
        for legacy in (False, True):
            current = ev.load_minn(ROOT / "MINN_data", tokens, model_family="D", legacy_protocol=legacy)
            self.assertEqual(set(current["fixed"]), set(d.COMMON_BASE_EXCHANGES))
            self.assertEqual(set(current["fixed"].values()), {10})
            historical = ev.load_minn(ROOT / "MINN_data", tokens,
                                      input_names=archived.build_input_columns(),
                                      model_family="D", legacy_protocol=legacy)
            self.assertEqual(set(historical["fixed"].values()), {50})
            self.assertEqual(set(historical["fixed"]),
                             set(ev.COMMON_BASE_EXCHANGES + ev.B_ONLY_BASE_EXCHANGES))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old_d.pth"
            torch.save(dict(model_state_dict={}, config=dict(d_model=4, n_heads=2,
                n_layers=1, d_ff=8, dropout=0, vocab_size=len(tokens)),
                data_info=dict(input_cols=archived.build_input_columns(), output_cols=tokens)), path)
            with self.assertRaisesRegex(ValueError, "Checkpoint input order does not match model D"):
                ev.load_reservoir(path, "cpu", model_family="D")


if __name__ == "__main__":
    unittest.main()

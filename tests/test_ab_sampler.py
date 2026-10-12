"""Revised B basal caps and 40-input A-union-B sampling checks."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import generate_ecoli_iML1515_A_data as a
import generate_ecoli_iML1515_B_data as b
import generate_ecoli_iML1515_AB_union_data as ab


def arguments(module, argv=()):
    with patch.object(sys, "argv", ["generator", *argv]):
        return module.parse_args()


class ABSamplerTest(unittest.TestCase):
    def test_uniform_context_caps(self):
        ba, aba = arguments(b), arguments(ab)
        self.assertEqual((ba.ethanol_cap_max, ba.acetate_cap_max), (2, 8))
        self.assertEqual((aba.b_ethanol_cap_max, aba.b_acetate_cap_max), (2, 8))
        for module in (b, ab):
            rng = np.random.default_rng(9)
            for minimum, maximum in ((1, 15), (0, 20), (0, 2), (0, 8)):
                values = np.array([module.random_uniform_cap(rng, minimum, maximum) for _ in range(10000)])
                self.assertTrue(((values >= minimum) & (values <= maximum)).all())
                normalized = (values - minimum) / (maximum - minimum)
                self.assertAlmostEqual(float(normalized.mean()), 0.5, delta=0.01)
                self.assertAlmostEqual(float(normalized.var()), 1 / 12, delta=0.003)
                self.assertTrue(np.any(values != np.floor(values)))

    def test_b_oxygen_matches_a(self):
        aa, ba, aba = arguments(a), arguments(b), arguments(ab)
        self.assertEqual((ba.oxygen_cap_min, ba.oxygen_cap_max),
                         (aa.oxygen_rate_min, aa.oxygen_rate_max))
        self.assertEqual((aba.b_oxygen_cap_min, aba.b_oxygen_cap_max),
                         (aa.oxygen_rate_min, aa.oxygen_rate_max))
        self.assertEqual((ba.co2_cap_min, ba.co2_cap_max), (0, 20))
        self.assertEqual((aba.b_co2_cap_min, aba.b_co2_cap_max), (0, 20))
        reference_rng = np.random.default_rng(9)
        actual_rng = np.random.default_rng(9)
        for _ in range(1000):
            self.assertEqual(b.random_loguniform_cap(actual_rng, 1, 25),
                             a.random_rate(reference_rng, 1, 25, log_uniform=True))

    def test_a_ranges_and_loguniform_draws(self):
        aa, args = arguments(a), arguments(ab)
        for ab_name, a_name in (("a_carbon_rate_min", "carbon_rate_min"),
                                ("a_carbon_rate_max", "carbon_rate_max"),
                                ("a_oxygen_rate_min", "oxygen_rate_min"),
                                ("a_oxygen_rate_max", "oxygen_rate_max")):
            self.assertEqual(getattr(args, ab_name), getattr(aa, a_name))
        rng = np.random.default_rng(9)
        for low, high in ((0.05, 10), (1, 25)):
            rates = np.array([ab.random_rate(rng, low, high) for _ in range(20000)])
            self.assertTrue(((rates >= low) & (rates <= high)).all())
            self.assertAlmostEqual(float(np.log(rates).mean()), np.log(low * high) / 2, delta=0.04)
            self.assertAlmostEqual(float(np.log(rates).var()), np.log(high / low) ** 2 / 12, delta=0.08)
        for flag in ("--a-carbon-rate-min", "--a-oxygen-rate-min"):
            with self.assertRaises(ValueError):
                ab.validate_args(arguments(ab, [flag, "0"]))

    def test_end_to_end_generators(self):
        for module, prefix, width in ((b, "b_check", 26), (ab, "ab_check", 40)):
            with self.subTest(generator=module.__name__), tempfile.TemporaryDirectory() as folder:
                args = arguments(module)
                self.assertEqual(args.base_rate if module is b else args.b_base_rate, 10)
                observed = []
                original = module.generate_training_sample

                def capture(*pos, **kwargs):
                    result = original(*pos, **kwargs)
                    if result[0] is not None:
                        model = kwargs["model"]
                        basal = (b.BASE_EXCHANGES if module is b else ab.COMMON_BASE_EXCHANGES)
                        for ex in basal:
                            if ex not in b.SAMPLED_EXCHANGES:
                                self.assertEqual(model.reactions.get_by_id(ex).lower_bound, -10)
                        self.assertEqual(model.reactions.EX_cbl1_e.lower_bound, 0)
                        self.assertTrue(1 <= result[0]["EX_o2_e"] <= 25)
                        if module is b or kwargs["regime"] == "b":
                            for ex, maximum in (("EX_co2_e", 20), ("EX_etoh_e", 2), ("EX_ac_e", 8)):
                                cap = result[0][ex]
                                self.assertTrue(0 <= cap <= maximum)
                                self.assertEqual(cap, round(cap, 2))
                                self.assertEqual(model.reactions.get_by_id(ex).upper_bound, cap)
                            glucose = result[0]["EX_glc__D_e"]
                            self.assertTrue(1 <= glucose <= 15)
                            self.assertEqual(glucose, round(glucose, 2))
                            self.assertEqual(model.reactions.EX_glc__D_e.lower_bound, -glucose)
                        if module is ab:
                            regime = kwargs["regime"]
                            observed.append(regime)
                            data = result[0]
                            if regime == "a":
                                self.assertTrue(1 <= data["EX_o2_e"] <= 25)
                                sources = [ex for ex in ab.AMN_CARBON_EXCHANGES if data[ex] > 0]
                                self.assertTrue(1 <= len(sources) <= 4)
                                for ex in sources:
                                    self.assertTrue(0.05 <= data[ex] <= 10)
                                for ex in ab.FIXED_CARBON_EXCHANGES + ab.AMINO_EXCHANGES:
                                    self.assertEqual(data[ex], 2.2)
                    return result

                argv = ["generator", "--n-samples", "10", "--seed", "9", "--data-dir", folder,
                        "--output-prefix", prefix]
                with patch.object(sys, "argv", argv), patch.object(module, "generate_training_sample", capture), contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                data = pd.read_csv(Path(folder) / f"{prefix}_10_samples.csv")
                self.assertEqual(data.shape, (10, width + 2712))
                self.assertTrue(np.isfinite(data.to_numpy()).all())
                self.assertNotIn("EX_cbl1_e", data.columns)
                self.assertTrue((data.EX_cbl1_e_flux >= -1e-8).all())
                for ex in ("EX_glc__D_e", "EX_co2_e", "EX_etoh_e", "EX_ac_e"):
                    self.assertTrue((data[ex] != np.floor(data[ex])).any())
                if module is ab:
                    self.assertEqual(observed.count("a"), 5)
                    self.assertEqual(observed.count("b"), 5)
                    self.assertEqual(list(data.columns[:40]), ab.build_input_columns())
                    self.assertNotIn("EX_cbl1_e", data.columns)
                    self.assertTrue((data.EX_cbl1_e_flux >= -1e-8).all())


if __name__ == "__main__":
    unittest.main()

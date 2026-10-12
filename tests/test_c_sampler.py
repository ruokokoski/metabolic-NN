"""Checks for the revised C mixture and mixed-medium bounds."""

import sys
import unittest
from unittest.mock import patch

import numpy as np

import generate_ecoli_iML1515_C_data as sampler


def parse_args(argv=()):
    with patch.object(sys, "argv", ["generator", *argv]):
        return sampler.parse_args()


class CSamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = parse_args()
        cls.model, cls.bounds, cls.outputs = sampler.load_generation_model(
            "models", cls.args.objective_reaction, 120,
        )

    def test_regime_frequencies_and_vocabulary(self):
        self.assertEqual((self.args.oxygen_rate_min, self.args.oxygen_rate_max), (1, 25))
        self.assertEqual((self.args.co2_secretion_cap_min, self.args.co2_secretion_cap_max), (0, 20))
        self.assertEqual((self.args.ethanol_secretion_cap_max, self.args.acetate_secretion_cap_max), (2, 8))
        names, probabilities = sampler.normalize_regime_weights(self.args)
        self.assertEqual(dict(zip(names, probabilities)),
                         {"minn": 0.4, "faure": 0.4, "mixed": 0.2})
        rng = np.random.default_rng(42)
        draws = [sampler.choose_regime(rng, names, probabilities) for _ in range(20000)]
        for name, expected in zip(names, probabilities):
            self.assertAlmostEqual(draws.count(name) / len(draws), expected, delta=0.015)
        inputs, _ = sampler.build_input_cols()
        self.assertEqual(len(inputs), 40)
        self.assertNotIn("EX_cbl1_e", inputs)
        included = parse_args(["--include-cbl1"])
        self.assertFalse(included.exclude_cbl1)
        self.assertEqual(len(sampler.build_input_cols(included.exclude_cbl1)[0]), 41)
        self.assertTrue(parse_args(["--exclude-cbl1"]).exclude_cbl1)

    def test_mixed_cardinality_and_loguniform_caps(self):
        rng = np.random.default_rng(9)
        counts = []
        rates = []
        identity_counts = dict.fromkeys(sampler.NON_CONTEXT_FAURE_CARBON_EXCHANGES, 0)
        _, basal = sampler.build_input_cols()
        # Capture configured bounds without repeated optimization.
        with patch.object(sampler, "set_secretion_cap"):
            for _ in range(6000):
                with patch.object(sampler, "set_uptake") as uptake:
                    sampler.apply_mixed_regime(self.model, {}, rng, basal, self.args)
                additions = [(c.args[2], c.args[3]) for c in uptake.call_args_list
                             if c.args[2] in identity_counts]
                counts.append(len(additions))
                self.assertEqual(len(additions), len({ex for ex, _ in additions}))
                for ex, rate in additions:
                    identity_counts[ex] += 1
                    rates.append(rate)
        for k in (1, 2, 3):
            self.assertAlmostEqual(counts.count(k) / len(counts), 1 / 3, delta=0.025)
        for count in identity_counts.values():
            self.assertAlmostEqual(count / sum(counts), 1 / 9, delta=0.015)
        self.assertTrue(all(0.05 <= rate <= 10 for rate in rates))
        self.assertAlmostEqual(float(np.log(rates).mean()), np.log(0.5) / 2, delta=0.06)
        self.assertAlmostEqual(float(np.log(rates).var()), np.log(200) ** 2 / 12, delta=0.1)

    def test_actual_pfba_regimes_and_bounds(self):
        inputs, basal = sampler.build_input_cols()
        for regime in ("faure", "minn", "mixed"):
            with self.subTest(regime=regime):
                data, status, actual_regime = sampler.generate_training_sample(
                    self.model, np.random.default_rng(9), self.bounds, self.outputs,
                    inputs, basal, [regime], np.array([1.0]), self.args,
                )
                self.assertEqual(status, "optimal")
                self.assertEqual(actual_regime, regime)
                for ex in basal:
                    self.assertEqual(self.model.reactions.get_by_id(ex).lower_bound, -10)
                self.assertEqual(self.model.reactions.EX_cbl1_e.lower_bound, 0)
                if regime == "faure":
                    self.assertEqual(data["EX_glc__D_e"], 0)
                    self.assertEqual(self.model.reactions.EX_co2_e.lower_bound, -10)
                    self.assertTrue(1 <= data["EX_o2_e"] <= 25)
                    for ex in sampler.FIXED_CARBON_EXCHANGES + sampler.AMINO_EXCHANGES:
                        self.assertEqual(data[ex], 2.2)
                    sources = [ex for ex in sampler.FAURE_CARBON_EXCHANGES if data[ex] > 0]
                    self.assertTrue(1 <= len(sources) <= 4)
                else:
                    self.assertTrue(1 <= data["EX_glc__D_e"] <= 15)
                    self.assertEqual(data["EX_glc__D_e"], round(data["EX_glc__D_e"], 2))
                    self.assertEqual(self.model.reactions.EX_glc__D_e.lower_bound, -data["EX_glc__D_e"])
                    self.assertTrue(1 <= data["EX_o2_e"] <= 25)
                    for ex, maximum in (("EX_co2_e", 20), ("EX_etoh_e", 2), ("EX_ac_e", 8)):
                        reaction = self.model.reactions.get_by_id(ex)
                        self.assertEqual(reaction.lower_bound, 0)
                        self.assertEqual(reaction.upper_bound, data[ex])
                        self.assertTrue(0 <= data[ex] <= maximum)
                        self.assertEqual(data[ex], round(data[ex], 2))
                    for ex in sampler.FIXED_CARBON_EXCHANGES + sampler.AMINO_EXCHANGES:
                        self.assertEqual(self.model.reactions.get_by_id(ex).lower_bound, 0)
                    sources = [ex for ex in sampler.NON_CONTEXT_FAURE_CARBON_EXCHANGES if data[ex] > 0]
                    if regime == "mixed":
                        self.assertTrue(1 <= len(sources) <= 3)
                    else:
                        self.assertEqual(sources, [])
                for ex in sources:
                    self.assertTrue(0.05 <= data[ex] <= 10)
                    self.assertEqual(self.model.reactions.get_by_id(ex).lower_bound, -data[ex])
                biomass = data[self.args.objective_reaction + "_flux"]
                optimum = self.model.slim_optimize()
                self.assertAlmostEqual(biomass, 0.999 * optimum, delta=1e-6)
                fluxes = np.array([data[rxn + "_flux"] for rxn in self.outputs])
                self.assertTrue(np.isfinite(fluxes).all())


if __name__ == "__main__":
    unittest.main()

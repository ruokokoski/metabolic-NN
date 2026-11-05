"""Analytic and output-integrity checks for the Euclidean fluxomics projection."""

import json
from pathlib import Path
import tempfile
import unittest

from cobra import Metabolite, Model, Reaction
from cobra.io import write_sbml_model

from scripts import fit_minn_fluxomics_euclidean as fit


def toy_model():
    model = Model("euclidean_test")
    a, b = Metabolite("a_c", compartment="c"), Metabolite("b_c", compartment="c")
    uptake = Reaction("EX_glc__D_e", lower_bound=-10, upper_bound=0)
    uptake.add_metabolites({a: -1})
    middle = Reaction("MID", lower_bound=0, upper_bound=10)
    middle.add_metabolites({a: -1, b: 1})
    biomass = Reaction(fit.DEFAULT_OBJECTIVE_REACTION, lower_bound=0, upper_bound=10)
    biomass.add_metabolites({b: -1})
    model.add_reactions([uptake, middle, biomass])
    model.objective = biomass
    return model


class EuclideanFitTest(unittest.TestCase):
    def setUp(self):
        self.model = toy_model()
        self.columns = ["experiment", "R_EX_glc__D_e_rev", "R_MID", fit.DEFAULT_BIOMASS_COLUMN]
        self.row = dict(zip(self.columns, ["sample", "0", "0", "6"]))
        lookup = fit.reaction_lookup(self.model)
        self.mappings = {
            name: fit.map_flux_column(name, lookup, fit.DEFAULT_BIOMASS_COLUMN,
                                      fit.DEFAULT_OBJECTIVE_REACTION)
            for name in self.columns[1:]
        }

    def test_unweighted_euclidean_projection_and_reverse_sign(self):
        # All three magnitudes must agree: squared error is 3*(v-2)^2 + 24.
        # Absolute error instead has its unique minimum at v=0.
        args = fit.build_arg_parser().parse_args(["--biomass-mode", "free"])
        fitted, diagnostic = fit.solve_sample(self.model, self.row, self.mappings,
                                               self.columns, args)
        self.assertEqual(diagnostic["status"], "optimal", diagnostic)
        for name in self.columns[1:]:
            self.assertAlmostEqual(fitted[name], 2.0, delta=0.001)
        self.assertAlmostEqual(diagnostic["squared_distance"], 24.0, delta=1e-5)
        self.assertLessEqual(diagnostic["optimality_gap"],
                             args.absolute_gap + args.relative_gap * 24.00001)
        self.assertLessEqual(diagnostic["max_mass_balance_error"], args.feasibility_tolerance)
        self.assertLessEqual(diagnostic["max_bound_violation"], args.feasibility_tolerance)

    def test_fixed_biomass_and_model_context_restoration(self):
        args = fit.build_arg_parser().parse_args([])
        before = [(r.id, r.bounds) for r in self.model.reactions]
        fitted, diagnostic = fit.solve_sample(self.model, self.row, self.mappings,
                                               self.columns, args)
        self.assertEqual(diagnostic["status"], "optimal", diagnostic)
        self.assertEqual(fitted[fit.DEFAULT_BIOMASS_COLUMN], 6.0)
        self.assertAlmostEqual(diagnostic["squared_distance"], 72.0)
        self.assertEqual([(r.id, r.bounds) for r in self.model.reactions], before)
        self.assertEqual(len(self.model.solver.variables), 6)

    def test_default_glucose_cap_allows_measurements_above_sbml_cap(self):
        self.model.reactions.get_by_id("MID").upper_bound = 1000
        self.model.reactions.get_by_id(fit.DEFAULT_OBJECTIVE_REACTION).upper_bound = 1000
        row = dict(zip(self.columns, ["high_glucose", "13.34", "13.34", "13.34"]))
        args = fit.build_arg_parser().parse_args([])
        fitted, diagnostic = fit.solve_sample(self.model, row, self.mappings,
                                               self.columns, args)
        self.assertEqual(diagnostic["status"], "optimal", diagnostic)
        self.assertAlmostEqual(fitted["R_EX_glc__D_e_rev"], 13.34)
        self.assertAlmostEqual(diagnostic["squared_distance"], 0.0)
        self.assertEqual(self.model.reactions.get_by_id("EX_glc__D_e").lower_bound, -10)

    def test_explicit_glucose_cap_is_respected(self):
        self.model.reactions.get_by_id("MID").upper_bound = 1000
        self.model.reactions.get_by_id(fit.DEFAULT_OBJECTIVE_REACTION).upper_bound = 1000
        row = dict(zip(self.columns, ["high_glucose", "13.34", "13.34", "13.34"]))
        args = fit.build_arg_parser().parse_args([
            "--biomass-mode", "free", "--glucose-uptake-cap", "12",
        ])
        fitted, diagnostic = fit.solve_sample(self.model, row, self.mappings,
                                               self.columns, args)
        self.assertEqual(diagnostic["status"], "optimal", diagnostic)
        self.assertAlmostEqual(fitted["R_EX_glc__D_e_rev"], 12.0, delta=1e-5)

    def test_incomplete_fit_never_writes_fitted_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, sbml = root / "source.csv", root / "fit.csv", root / "model.xml"
            impossible = dict(self.row, experiment="outside_bounds")
            impossible[fit.DEFAULT_BIOMASS_COLUMN] = "20"
            fit.write_csv_dicts(source, self.columns, [self.row, impossible])
            write_sbml_model(self.model, str(sbml))
            args = fit.build_arg_parser().parse_args([
                "--source-csv", str(source), "--output-csv", str(output),
                "--model-sbml", str(sbml),
            ])
            self.assertEqual(fit.run_with_args(args), 1)
            self.assertFalse(output.exists())
            metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertFalse(metadata["fitted_csv_written"])
            self.assertEqual(metadata["successful_conditions"], 1)
            self.assertEqual(metadata["total_conditions"], 2)
            self.assertEqual(metadata["arguments"]["glucose_uptake_cap"], 1000)
            self.assertEqual(metadata["exchange_bound_overrides"]["EX_glc__D_e"],
                             {"sbml_lower_bound": -10, "fit_lower_bound": -1000})

    def test_existing_output_is_preserved_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.csv", root / "fit.csv"
            source.write_text("experiment,x\nsample,1\n")
            output.write_text("existing result\n")
            args = fit.build_arg_parser().parse_args([
                "--source-csv", str(source), "--output-csv", str(output),
            ])
            with self.assertRaises(FileExistsError):
                fit.run_with_args(args)
            self.assertEqual(output.read_text(), "existing result\n")


if __name__ == "__main__":
    unittest.main()

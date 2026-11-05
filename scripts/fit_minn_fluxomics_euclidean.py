#!/usr/bin/env python
"""Project unfitted MINN flux measurements into the iML1515 feasible space.

Minimize sum_j (sign_j * v_j - measured_j)**2 over the mapped measured
fluxes, subject to steady state and the SBML reaction bounds. All errors have
equal weight. Biomass is fixed to its measured value by default, matching the
earlier local fitter; --biomass-mode free includes it as an adjustable target.
Glucose and oxygen can move within the model bounds. No knockout constraints
or growth-maximization objective are added.

The installed GLPK solver handles linear objectives only. Successive tangent
planes underestimate each squared error, giving a sequence of LP lower bounds
on the convex quadratic optimum. A feasible solution supplies an upper bound;
the script accepts a fit only when their gap meets the requested tolerance.
This avoids adding a quadratic-solver dependency or changing the L2 objective.
If the floating-point LP fails these checks, the condition is retried with
GLPK's exact-arithmetic LP interface. Each LP solve has an explicit time limit.

This is a documented iML1515 sensitivity fit, not an exact reproduction of the
unreleased historical MINN fitting implementation. It does not train model B.
Run from the repository root, for example:

    python scripts/fit_minn_fluxomics_euclidean.py

The fitted CSV preserves the source column/condition order. Diagnostics and
JSON provenance accompany it. An incomplete fit never replaces the output CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

try:
    from .fit_minn_fluxomics_minn_like import (
        DEFAULT_BIOMASS_COLUMN,
        DEFAULT_MODEL_SBML,
        DEFAULT_OBJECTIVE_REACTION,
        DEFAULT_SOURCE_CSV,
        apply_direction_bound,
        directional_target,
        map_flux_column,
        reaction_lookup,
        read_csv_dicts,
        write_csv_dicts,
    )
except ImportError:  # Direct execution puts scripts/ on the import path.
    from fit_minn_fluxomics_minn_like import (
        DEFAULT_BIOMASS_COLUMN,
        DEFAULT_MODEL_SBML,
        DEFAULT_OBJECTIVE_REACTION,
        DEFAULT_SOURCE_CSV,
        apply_direction_bound,
        directional_target,
        map_flux_column,
        reaction_lookup,
        read_csv_dicts,
        write_csv_dicts,
    )


DEFAULT_OUTPUT_CSV = Path("MINN_data/fluxomics_iML1515_euclidean_fit.csv")


def solve_sample(base_model, row, mappings, fieldnames, args):
    """Return one feasible, gap-certified projection and its diagnostics."""
    diagnostic = {
        "experiment": row["experiment"],
        "status": "error",
        "iterations": 0,
        "squared_distance": "",
        "euclidean_distance": "",
        "lower_bound": "",
        "optimality_gap": "",
        "max_mass_balance_error": "",
        "max_bound_violation": "",
        "solver": base_model.solver.interface.__name__,
        "fallback_reason": "",
        "message": "",
    }
    try:
        with base_model as model:
            if args.enforce_split_directions:
                for mapping in mappings.values():
                    apply_direction_bound(model, mapping)

            if args.biomass_mode == "fixed":
                mapping = mappings[args.biomass_column]
                reaction = model.reactions.get_by_id(mapping.reaction_id)
                value = directional_target(row, args.biomass_column) / mapping.sign
                if not reaction.lower_bound <= value <= reaction.upper_bound:
                    raise ValueError("Measured biomass is outside the model bounds.")
                reaction.bounds = (value, value)

            expressions = []
            squares = []
            for index, (column, mapping) in enumerate(mappings.items()):
                reaction = model.reactions.get_by_id(mapping.reaction_id)
                error = mapping.sign * reaction.flux_expression - directional_target(row, column)
                square = model.problem.Variable(f"euclidean_square_{index}", lb=0)
                model.add_cons_vars(square)
                expressions.append(error)
                squares.append(square)

            model.objective = model.problem.Objective(sum(squares), direction="min")
            best_distance = math.inf
            best_fluxes = None
            for iteration in range(1, args.max_iterations + 1):
                solution = model.optimize()
                diagnostic["iterations"] = iteration
                if solution.status != "optimal":
                    raise RuntimeError(f"LP status: {solution.status}")
                lower_bound = float(solution.objective_value)
                errors = [
                    mapping.sign * float(solution.fluxes[mapping.reaction_id])
                    - directional_target(row, column)
                    for column, mapping in mappings.items()
                ]
                squared_distance = math.fsum(error * error for error in errors)
                if not math.isfinite(squared_distance) or not math.isfinite(lower_bound):
                    raise RuntimeError("Nonfinite objective or flux measurement.")
                if squared_distance < best_distance:
                    if not all(math.isfinite(float(value)) for value in solution.fluxes):
                        raise RuntimeError("The full flux solution contains nonfinite values.")
                    best_distance = squared_distance
                    best_fluxes = solution.fluxes.copy()
                gap = best_distance - lower_bound
                tolerance = args.absolute_gap + args.relative_gap * best_distance
                diagnostic.update(
                    squared_distance=best_distance,
                    euclidean_distance=math.sqrt(best_distance),
                    lower_bound=lower_bound,
                    optimality_gap=gap,
                )
                if gap < -tolerance:
                    raise RuntimeError("LP lower bound exceeds the feasible squared error.")
                if gap <= tolerance:
                    break

                # e^2 >= 2*a*e - a^2 for every e and tangent point a.
                cuts = [
                    model.problem.Constraint(
                        square - 2.0 * error * expression,
                        lb=-error * error,
                        name=f"euclidean_cut_{iteration}_{index}",
                    )
                    for index, (square, expression, error) in enumerate(
                        zip(squares, expressions, errors)
                    )
                    if error * error - float(square.primal) > 0
                ]
                if not cuts:
                    raise RuntimeError("LP refinement stalled before closing the objective gap.")
                model.add_cons_vars(cuts)
            else:
                raise RuntimeError("Maximum LP refinements reached before convergence.")

            mass_balance_error = max(
                (
                    abs(math.fsum(
                        reaction.metabolites[metabolite] * float(best_fluxes[reaction.id])
                        for reaction in metabolite.reactions
                    ))
                    for metabolite in model.metabolites
                ),
                default=0.0,
            )
            bound_violation = max(
                (
                    max(
                        reaction.lower_bound - float(best_fluxes[reaction.id]),
                        float(best_fluxes[reaction.id]) - reaction.upper_bound,
                        0.0,
                    )
                    for reaction in model.reactions
                ),
                default=0.0,
            )
            diagnostic.update(
                max_mass_balance_error=mass_balance_error,
                max_bound_violation=bound_violation,
            )
            if max(mass_balance_error, bound_violation) > args.feasibility_tolerance:
                raise RuntimeError("The fitted full flux vector fails the feasibility check.")

            fitted = {"experiment": row["experiment"]}
            for column in fieldnames:
                if column != "experiment":
                    mapping = mappings[column]
                    fitted[column] = mapping.sign * float(best_fluxes[mapping.reaction_id])
            diagnostic.update(status="optimal", message="gap and feasibility checks passed")
            return fitted, diagnostic
    except Exception as exc:
        diagnostic["message"] = str(exc)
        return None, diagnostic


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--model-sbml", type=Path, default=DEFAULT_MODEL_SBML)
    parser.add_argument("--objective-reaction", default=DEFAULT_OBJECTIVE_REACTION,
                        help="iML1515 reaction used to map the biomass column; not maximized.")
    parser.add_argument("--biomass-column", default=DEFAULT_BIOMASS_COLUMN)
    parser.add_argument("--biomass-mode", choices=("fixed", "free"), default="fixed")
    parser.add_argument("--no-enforce-split-directions", dest="enforce_split_directions",
                        action="store_false", help="Allow the mapped net flux to change direction.")
    parser.set_defaults(enforce_split_directions=True)
    parser.add_argument("--absolute-gap", type=float, default=1e-6,
                        help="Absolute tolerance on the sum-of-squares optimality gap.")
    parser.add_argument("--relative-gap", type=float, default=1e-7)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-6)
    parser.add_argument("--solver-tolerance", type=float, default=1e-12,
                        help="COBRA LP feasibility/optimality tolerance for this new fit.")
    parser.add_argument("--solver", choices=("glpk", "glpk_exact"), default="glpk",
                        help="Primary LP solver; failed float-LP fits retry with glpk_exact.")
    parser.add_argument("--solver-timeout", type=int, default=10,
                        help="Time limit in seconds for each LP solve, including exact retries.")
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing fit or its sidecar files.")
    return parser


def _write_csv_atomic(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    os.close(descriptor)
    try:
        write_csv_dicts(Path(temporary), fieldnames, rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_with_args(args: argparse.Namespace) -> int:
    import cobra
    from cobra.io import read_sbml_model

    for name in ("absolute_gap", "relative_gap", "feasibility_tolerance", "solver_tolerance"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if (args.absolute_gap + args.relative_gap == 0 or args.feasibility_tolerance == 0
            or not 0 < args.solver_tolerance < 1):
        raise ValueError("Gap and feasibility tolerances must be positive; solver tolerance must be in (0, 1).")
    if args.max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if args.solver_timeout < 1:
        raise ValueError("solver_timeout must be positive.")
    diagnostics_path = args.output_csv.with_suffix(".diagnostics.csv")
    metadata_path = args.output_csv.with_suffix(".metadata.json")
    inputs = {args.source_csv.resolve(), args.model_sbml.resolve()}
    for path in (args.output_csv, diagnostics_path, metadata_path):
        if path.resolve() in inputs:
            raise ValueError("Output files must differ from the source CSV and SBML model.")
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite explicitly.")

    fieldnames, rows = read_csv_dicts(args.source_csv)
    if "experiment" not in fieldnames or len(fieldnames) != len(set(fieldnames)):
        raise ValueError("Source needs an experiment column and unique column names.")
    experiments = [row["experiment"] for row in rows]
    if not experiments or any(not value for value in experiments) or len(experiments) != len(set(experiments)):
        raise ValueError("Source needs nonempty, unique experiment IDs and at least one row.")
    model = read_sbml_model(str(args.model_sbml))
    model.solver = args.solver
    model.tolerance = args.solver_tolerance
    model.solver.configuration.timeout = args.solver_timeout
    lookup = reaction_lookup(model)
    mappings = {}
    for column in fieldnames:
        if column == "experiment":
            continue
        mapping = map_flux_column(column, lookup, args.biomass_column, args.objective_reaction)
        if mapping is None:
            raise ValueError(f"Unmapped measured flux column: {column}")
        mappings[column] = mapping
        for row in rows:
            directional_target(row, column)
    if not mappings or (args.biomass_mode == "fixed" and args.biomass_column not in mappings):
        raise ValueError("No measured fluxes, or the fixed biomass column is missing.")

    fitted_rows, diagnostics = [], []
    exact_model = None
    for row in rows:
        fitted, diagnostic = solve_sample(model, row, mappings, fieldnames, args)
        if fitted is None and diagnostic["iterations"] > 0 and args.solver != "glpk_exact":
            reason = diagnostic["message"]
            print(f"{row['experiment']}: retrying exact LP after {reason}", flush=True)
            if exact_model is None:
                exact_model = model.copy()
                exact_model.solver = "glpk_exact"
                exact_model.tolerance = args.solver_tolerance
                exact_model.solver.configuration.timeout = args.solver_timeout
            fitted, diagnostic = solve_sample(exact_model, row, mappings, fieldnames, args)
            diagnostic["fallback_reason"] = reason
        diagnostics.append(diagnostic)
        if fitted is not None:
            fitted_rows.append(fitted)
        print(f"{row['experiment']}: {diagnostic['status']}, "
              f"SSE={diagnostic['squared_distance']}, gap={diagnostic['optimality_gap']}, "
              f"iterations={diagnostic['iterations']}", flush=True)
        if fitted is None:
            print(f"  {diagnostic['message']}", flush=True)

    complete = len(fitted_rows) == len(rows)
    metadata = {
        "objective": "unweighted sum of squared deviations over mapped measured fluxes",
        "algorithm": "LP tangent outer approximation with feasible upper and LP lower bounds",
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "source_sha256": hashlib.sha256(args.source_csv.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model_sbml.read_bytes()).hexdigest(),
        "python_version": sys.version,
        "cobra_version": cobra.__version__,
        "solver": model.solver.interface.__name__,
        "fallback_solver": "glpk_exact" if exact_model is not None else None,
        "solver_tolerance": model.tolerance,
        "flux_columns": list(mappings),
        "mappings": {column: vars(mapping) for column, mapping in mappings.items()},
        "experiments": experiments,
        "successful_conditions": len(fitted_rows),
        "total_conditions": len(rows),
        "fitted_csv_written": complete,
        "knockout_constraints": False,
        "growth_maximization": False,
    }
    _write_csv_atomic(diagnostics_path, list(diagnostics[0]), diagnostics)
    if complete:
        _write_csv_atomic(args.output_csv, fieldnames, fitted_rows)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not complete:
        print(f"Solved {len(fitted_rows)}/{len(rows)}; fitted CSV was not written.")
        return 1
    print(f"Saved {len(rows)} fitted conditions to {args.output_csv}")
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run_with_args(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

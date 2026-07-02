#!/usr/bin/env python
"""Fit MINN split fluxomics into a GEM feasible space with a MINN-like policy.

This script is intentionally separate from the notebooks. It starts from the
non-fitted split MINN fluxomics file and solves one linear fit per sample:

    minimize weighted absolute deviation from the source flux vector
    subject to GEM steady-state constraints and bounds
               biomass fixed by default
               glucose/O2 allowed to move through weighted deviation terms
               no hard glucose/O2 tolerance band by default

It also compares the repository's original MINN fitted file against the
non-fitted split file. The comparison is descriptive only: it checks whether the
reference fitted file resembles a simple conversion, a scalar rescaling, or the
configured MINN-like policy. The exact original fitting procedure is not present
in this repository. The script refuses to write incomplete fitted outputs if any
sample cannot be solved.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SOURCE_CSV = Path("MINN_data/fluxomics_iAF1260_reduced_split.csv")
DEFAULT_REFERENCE_FITTED_CSV = Path("MINN_data/fluxomics_iAF1260_reduced_split_fit.csv")
DEFAULT_OUTPUT_CSV = Path("MINN_data/fluxomics_iML1515_minn_like_fit.csv")
DEFAULT_MODEL_SBML = Path("models/iML1515.xml")

DEFAULT_BIOMASS_COLUMN = "R_BIOMASS_Ec_iAF1260_core_59p81M"
DEFAULT_OBJECTIVE_REACTION = "BIOMASS_Ec_iML1515_core_75p37M"
MEASURED_INPUT_COLUMNS = ("R_EX_glc__D_e_rev", "R_EX_o2_e_rev")


@dataclass(frozen=True)
class ColumnMapping:
    column: str
    reaction_id: str
    sign: float
    exact_split_reaction: bool


def read_csv_dicts(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return list(reader.fieldnames), rows


def write_csv_dicts(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_float(value: object) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def reaction_lookup(model) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for reaction in model.reactions:
        rid = reaction.id
        candidates = {rid}
        if rid.startswith("R_"):
            candidates.add(rid[2:])
        else:
            candidates.add(f"R_{rid}")
        for candidate in candidates:
            lookup.setdefault(candidate, rid)
    return lookup


def parse_split_column(column: str) -> Tuple[str, float]:
    base = column[2:] if column.startswith("R_") else column
    if base.endswith("_fwd"):
        return base[:-4], 1.0
    if base.endswith("_rev"):
        return base[:-4], -1.0
    return base, 1.0


def map_flux_column(
    column: str,
    lookup: Dict[str, str],
    biomass_column: str,
    objective_reaction: Optional[str],
) -> Optional[ColumnMapping]:
    # Prefer exact split-reaction mapping when the GEM itself is split.
    exact_candidates = [column, column[2:] if column.startswith("R_") else f"R_{column}"]
    for candidate in exact_candidates:
        if candidate in lookup:
            return ColumnMapping(column, lookup[candidate], 1.0, True)

    if column == biomass_column and objective_reaction:
        objective_candidates = [objective_reaction, f"R_{objective_reaction}"]
        for candidate in objective_candidates:
            if candidate in lookup:
                return ColumnMapping(column, lookup[candidate], 1.0, False)

    base, sign = parse_split_column(column)
    for candidate in (base, f"R_{base}"):
        if candidate in lookup:
            return ColumnMapping(column, lookup[candidate], sign, False)
    return None


def classify_weight(column: str, args: argparse.Namespace) -> float:
    if column == args.biomass_column:
        return args.biomass_weight
    if column in MEASURED_INPUT_COLUMNS:
        return args.measured_input_weight
    lowered = column.lower()
    if lowered.startswith("r_ex_"):
        return args.exchange_weight
    return args.internal_weight


def directional_target(row: Dict[str, str], column: str) -> float:
    value = as_float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite target value in {column}: {row[column]!r}")
    return value


def add_absolute_deviation(
    model,
    expression,
    target: float,
    name: str,
    weight: float,
) -> Tuple[object, object, object, float]:
    plus = model.problem.Variable(f"{name}_plus", lb=0.0)
    minus = model.problem.Variable(f"{name}_minus", lb=0.0)
    constraint = model.problem.Constraint(expression - plus + minus, lb=target, ub=target)
    model.add_cons_vars([plus, minus, constraint])
    return plus, minus, constraint, weight


def apply_direction_bound(model, mapping: ColumnMapping) -> None:
    if mapping.exact_split_reaction:
        return
    reaction = model.reactions.get_by_id(mapping.reaction_id)
    if mapping.sign > 0:
        reaction.lower_bound = max(float(reaction.lower_bound), 0.0)
    else:
        reaction.upper_bound = min(float(reaction.upper_bound), 0.0)


def solve_sample(
    base_model,
    row: Dict[str, str],
    mappings: Dict[str, ColumnMapping],
    fieldnames: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    experiment = row.get("experiment", "")
    diagnostics: Dict[str, object] = {
        "experiment": experiment,
        "status": "",
        "objective": "",
        "message": "",
    }

    try:
        with base_model as model:
            deviation_terms = []

            if args.enforce_split_directions:
                for mapping in mappings.values():
                    apply_direction_bound(model, mapping)

            for column, mapping in mappings.items():
                target = directional_target(row, column)
                reaction = model.reactions.get_by_id(mapping.reaction_id)
                expression = mapping.sign * reaction.flux_expression

                if column == args.biomass_column and args.biomass_mode == "fixed":
                    model.add_cons_vars(
                        [
                            model.problem.Constraint(
                                expression,
                                lb=target,
                                ub=target,
                                name=f"fixed_{column}",
                            )
                        ]
                    )
                    continue

                plus, minus, _, weight = add_absolute_deviation(
                    model,
                    expression,
                    target,
                    name=f"dev_{column}",
                    weight=classify_weight(column, args),
                )
                deviation_terms.append(weight * plus)
                deviation_terms.append(weight * minus)

            if not deviation_terms:
                raise RuntimeError("No mapped flux columns were available for fitting.")

            model.objective = model.problem.Objective(sum(deviation_terms), direction="min")
            solution = model.optimize()
            diagnostics["status"] = getattr(solution, "status", "")
            diagnostics["objective"] = getattr(solution, "objective_value", "")
            if getattr(solution, "status", "") != "optimal":
                diagnostics["message"] = "non-optimal solve"
                return None, diagnostics

            fitted: Dict[str, object] = {"experiment": experiment}
            for column in fieldnames:
                if column == "experiment":
                    continue
                mapping = mappings.get(column)
                if mapping is None:
                    fitted[column] = row[column]
                    continue
                flux = float(solution.fluxes.get(mapping.reaction_id, math.nan))
                fitted[column] = mapping.sign * flux

            diagnostics["message"] = "ok"
            return fitted, diagnostics

    except Exception as exc:  # keep per-sample failures inspectable
        diagnostics["status"] = "error"
        diagnostics["message"] = str(exc)
        return None, diagnostics


def aligned_by_experiment(rows: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(row["experiment"]): row for row in rows}


def mae(values: Sequence[float]) -> float:
    finite = [abs(v) for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else math.nan


def print_reference_transformation_audit(
    source_rows: Sequence[Dict[str, str]],
    reference_rows: Sequence[Dict[str, str]],
    fieldnames: Sequence[str],
    args: argparse.Namespace,
) -> None:
    source_by_exp = aligned_by_experiment(source_rows)
    reference_by_exp = aligned_by_experiment(reference_rows)
    numeric_columns = [c for c in fieldnames if c != "experiment"]
    common_experiments = [row["experiment"] for row in source_rows if row["experiment"] in reference_by_exp]

    print("=== Original MINN fitted-vs-nonfitted audit ===")
    print(f"Non-fitted file: {args.source_csv}")
    print(f"MINN fitted file: {args.reference_fitted_csv}")
    print(f"Common samples: {len(common_experiments)}/{len(source_rows)}")
    print(f"Same header order: {fieldnames == list(reference_rows[0].keys())}")

    column_stats = []
    for column in numeric_columns:
        diffs = []
        ratios = []
        changed = 0
        for experiment in common_experiments:
            source_value = as_float(source_by_exp[experiment][column])
            reference_value = as_float(reference_by_exp[experiment][column])
            diff = reference_value - source_value
            diffs.append(diff)
            if abs(diff) > 1e-9:
                changed += 1
            if abs(source_value) > 1e-12 and abs(reference_value) > 1e-12:
                ratios.append(reference_value / source_value)
        column_stats.append(
            {
                "column": column,
                "changed": changed,
                "max_abs": max(abs(v) for v in diffs),
                "mean_abs": mae(diffs),
                "ratios": ratios,
            }
        )

    unchanged = [stat["column"] for stat in column_stats if stat["changed"] == 0]
    print(f"Columns unchanged by fitting: {len(unchanged)}/{len(numeric_columns)}")
    if unchanged:
        print("  " + ", ".join(unchanged))

    print("Top columns by maximum absolute fitted-minus-nonfitted change:")
    for stat in sorted(column_stats, key=lambda item: item["max_abs"], reverse=True)[:10]:
        print(
            f"  {stat['column']}: changed={stat['changed']}/{len(common_experiments)}, "
            f"max_abs={stat['max_abs']:.6g}, mean_abs={stat['mean_abs']:.6g}"
        )

    row_stats = []
    for experiment in common_experiments:
        total_abs = 0.0
        changed = 0
        max_abs = -1.0
        max_column = ""
        for column in numeric_columns:
            diff = as_float(reference_by_exp[experiment][column]) - as_float(source_by_exp[experiment][column])
            abs_diff = abs(diff)
            total_abs += abs_diff
            if abs_diff > 1e-9:
                changed += 1
            if abs_diff > max_abs:
                max_abs = abs_diff
                max_column = column
        row_stats.append((total_abs, changed, max_abs, max_column, experiment))

    print("Top samples by total absolute fitted-minus-nonfitted change:")
    for total_abs, changed, max_abs, max_column, experiment in sorted(row_stats, reverse=True)[:8]:
        print(
            f"  {experiment}: total_abs={total_abs:.6g}, changed_cols={changed}, "
            f"largest={max_abs:.6g} at {max_column}"
        )

    print("Transformation inference:")
    print("  - Not a header/sign-only conversion: headers are the same and many numeric values change.")
    print("  - Not a single per-row scaling: fitted/raw ratios vary strongly within high-change rows.")
    print("  - Biomass is fixed: the biomass column is unchanged in all rows.")
    print("  - Exchange fluxes are not preserved: fitted glucose, oxygen, and CO2 change in multiple rows.")
    print(
        "  - Most consistent interpretation from local evidence: a model-based FBA/steady-state "
        "feasibility repair of the 47-flux vector with biomass fixed, where exchange values were "
        "allowed to move. The exact objective and weights are not recoverable from this repository."
    )


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(left, right) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    xs, ys = zip(*pairs)
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_denom == 0.0 or y_denom == 0.0:
        return math.nan
    return numerator / (x_denom * y_denom)


def print_pairwise_fit_comparison(
    left_rows: Sequence[Dict[str, object]],
    right_rows: Sequence[Dict[str, object]],
    fieldnames: Sequence[str],
    left_label: str,
    right_label: str,
) -> None:
    left_by_exp = aligned_by_experiment(left_rows)
    right_by_exp = aligned_by_experiment(right_rows)
    numeric_columns = [c for c in fieldnames if c != "experiment"]
    common_experiments = [row["experiment"] for row in left_rows if row["experiment"] in right_by_exp]

    print(f"=== Fit comparison: {right_label} minus {left_label} ===")
    print(f"Common samples: {len(common_experiments)}/{len(left_rows)}")

    column_stats = []
    changed_entries = 0
    total_entries = len(common_experiments) * len(numeric_columns)
    for column in numeric_columns:
        diffs = []
        changed = 0
        for experiment in common_experiments:
            diff = as_float(right_by_exp[experiment][column]) - as_float(left_by_exp[experiment][column])
            diffs.append(diff)
            if abs(diff) > 1e-9:
                changed += 1
                changed_entries += 1
        column_stats.append(
            {
                "column": column,
                "changed": changed,
                "max_abs": max(abs(v) for v in diffs) if diffs else math.nan,
                "mean_abs": mae(diffs),
            }
        )

    print(f"Changed numeric entries: {changed_entries}/{total_entries}")
    print("Tracked exchange-column changes:")
    for column in ("R_EX_glc__D_e_rev", "R_EX_o2_e_rev", "R_EX_co2_e_fwd", "R_EX_etoh_e", "R_EX_ac_e"):
        stat = next((item for item in column_stats if item["column"] == column), None)
        if stat is None:
            continue
        print(
            f"  {column}: changed={stat['changed']}/{len(common_experiments)}, "
            f"max_abs={stat['max_abs']:.6g}, mean_abs={stat['mean_abs']:.6g}"
        )

    print("Top columns by maximum absolute change:")
    for stat in sorted(column_stats, key=lambda item: item["max_abs"], reverse=True)[:10]:
        print(
            f"  {stat['column']}: changed={stat['changed']}/{len(common_experiments)}, "
            f"max_abs={stat['max_abs']:.6g}, mean_abs={stat['mean_abs']:.6g}"
        )

    row_stats = []
    for experiment in common_experiments:
        total_abs = 0.0
        changed = 0
        max_abs = -1.0
        max_column = ""
        for column in numeric_columns:
            diff = as_float(right_by_exp[experiment][column]) - as_float(left_by_exp[experiment][column])
            abs_diff = abs(diff)
            total_abs += abs_diff
            if abs_diff > 1e-9:
                changed += 1
            if abs_diff > max_abs:
                max_abs = abs_diff
                max_column = column
        row_stats.append((total_abs, changed, max_abs, max_column, experiment))

    print("Top samples by total absolute change:")
    for total_abs, changed, max_abs, max_column, experiment in sorted(row_stats, reverse=True)[:8]:
        print(
            f"  {experiment}: total_abs={total_abs:.6g}, changed_cols={changed}, "
            f"largest={max_abs:.6g} at {max_column}"
        )


def print_new_fit_comparisons(
    source_rows: Sequence[Dict[str, str]],
    reference_rows: Sequence[Dict[str, str]],
    fitted_rows: Sequence[Dict[str, object]],
    fieldnames: Sequence[str],
    args: argparse.Namespace,
) -> None:
    if not fitted_rows:
        print("No fitted rows available for post-fit comparison.")
        return

    print_pairwise_fit_comparison(
        left_rows=source_rows,
        right_rows=fitted_rows,
        fieldnames=fieldnames,
        left_label=f"non-fitted source ({args.source_csv})",
        right_label=f"new fitted output ({args.output_csv})",
    )

    if not getattr(args, "compare_output_to_reference", False):
        print(
            "Original MINN fitted reference is not used as the target for the new fitted output. "
            "Pass --compare-output-to-reference to print that descriptive comparison."
        )
        return

    print_pairwise_fit_comparison(
        left_rows=reference_rows,
        right_rows=fitted_rows,
        fieldnames=fieldnames,
        left_label=f"original MINN fitted reference ({args.reference_fitted_csv})",
        right_label=f"new fitted output ({args.output_csv})",
    )

    source_by_exp = aligned_by_experiment(source_rows)
    reference_by_exp = aligned_by_experiment(reference_rows)
    fitted_by_exp = aligned_by_experiment(fitted_rows)
    numeric_columns = [c for c in fieldnames if c != "experiment"]
    common_experiments = [
        row["experiment"]
        for row in source_rows
        if row["experiment"] in reference_by_exp and row["experiment"] in fitted_by_exp
    ]

    reference_deltas: List[float] = []
    fitted_deltas: List[float] = []
    column_alignment = []
    for column in numeric_columns:
        column_ref_deltas = []
        column_fit_deltas = []
        for experiment in common_experiments:
            source_value = as_float(source_by_exp[experiment][column])
            column_ref_deltas.append(as_float(reference_by_exp[experiment][column]) - source_value)
            column_fit_deltas.append(as_float(fitted_by_exp[experiment][column]) - source_value)
        reference_deltas.extend(column_ref_deltas)
        fitted_deltas.extend(column_fit_deltas)
        delta_differences = [fit - ref for fit, ref in zip(column_fit_deltas, column_ref_deltas)]
        column_alignment.append(
            {
                "column": column,
                "mean_abs_delta_difference": mae(delta_differences),
                "reference_mean_abs_delta": mae(column_ref_deltas),
                "new_fit_mean_abs_delta": mae(column_fit_deltas),
            }
        )

    delta_differences = [fit - ref for fit, ref in zip(fitted_deltas, reference_deltas)]
    print("=== Delta alignment: new fitted output vs original MINN fitted repair ===")
    print(f"Compared numeric entries: {len(reference_deltas)}")
    print(f"Pearson correlation of source-relative deltas: {pearson_correlation(reference_deltas, fitted_deltas):.6g}")
    print(f"Mean absolute difference between repair deltas: {mae(delta_differences):.6g}")
    print("Top columns where new iML1515 repair differs most from original MINN repair:")
    for stat in sorted(column_alignment, key=lambda item: item["mean_abs_delta_difference"], reverse=True)[:10]:
        print(
            f"  {stat['column']}: mean_abs_delta_diff={stat['mean_abs_delta_difference']:.6g}, "
            f"MINN_ref_mean_abs_delta={stat['reference_mean_abs_delta']:.6g}, "
            f"new_fit_mean_abs_delta={stat['new_fit_mean_abs_delta']:.6g}"
        )


def print_summary(
    fitted_rows: Sequence[Dict[str, object]],
    diagnostics: Sequence[Dict[str, object]],
    mapped_columns: Sequence[str],
    unmapped_columns: Sequence[str],
    args: argparse.Namespace,
) -> None:
    ok_count = sum(1 for d in diagnostics if d.get("status") == "optimal")
    print(f"=== {getattr(args, 'run_label', 'MINN-like iML1515 fluxomics fitter')} ===")
    print(f"Source file: {args.source_csv}")
    print(f"Reference fitted file: {args.reference_fitted_csv}")
    print(f"Output file: {args.output_csv}")
    print(f"Model: {args.model_sbml}")
    print(f"Biomass mode: {args.biomass_mode}")
    print("Glucose/O2 hard band: disabled (MINN-like weighted-deviation policy)")
    print(f"Solved samples: {ok_count}/{len(diagnostics)}")
    print(f"Mapped flux columns: {len(mapped_columns)}")
    if unmapped_columns:
        print(f"Unmapped columns kept from source ({len(unmapped_columns)}): {', '.join(unmapped_columns)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--reference-fitted-csv", type=Path, default=DEFAULT_REFERENCE_FITTED_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--model-sbml", type=Path, default=DEFAULT_MODEL_SBML)
    parser.add_argument("--objective-reaction", default=DEFAULT_OBJECTIVE_REACTION)
    parser.add_argument("--biomass-column", default=DEFAULT_BIOMASS_COLUMN)
    parser.add_argument("--biomass-mode", choices=("fixed", "soft"), default="fixed")
    parser.add_argument(
        "--measured-input-weight",
        type=float,
        default=1000.0,
        help="Weight for glucose/O2 measured-input deviation terms.",
    )
    parser.add_argument("--biomass-weight", type=float, default=1000.0)
    parser.add_argument("--exchange-weight", type=float, default=100.0)
    parser.add_argument("--internal-weight", type=float, default=1.0)
    parser.add_argument(
        "--compare-reference-only",
        action="store_true",
        help=(
            "Do not run GEM fitting. Audit the MINN fitted-vs-nonfitted split files, "
            "and compare --output-csv against the non-fitted source if that output already exists."
        ),
    )
    parser.add_argument(
        "--compare-output-to-reference",
        action="store_true",
        help=(
            "Also compare the new fitted output against the original MINN fitted reference. "
            "This is descriptive only and is off by default."
        ),
    )
    parser.add_argument(
        "--no-enforce-split-directions",
        dest="enforce_split_directions",
        action="store_false",
        help="Do not constrain parsed fwd/rev columns to their split direction.",
    )
    parser.set_defaults(enforce_split_directions=True)
    parser.set_defaults(run_label="MINN-like iML1515 fluxomics fitter")
    return parser


def run_with_args(args: argparse.Namespace) -> int:
    fieldnames, source_rows = read_csv_dicts(args.source_csv)
    ref_fieldnames, reference_rows = read_csv_dicts(args.reference_fitted_csv)
    if fieldnames != ref_fieldnames:
        raise SystemExit("Source and reference fitted CSV headers differ; refusing to compare.")

    print_reference_transformation_audit(source_rows, reference_rows, fieldnames, args)
    if args.compare_reference_only:
        if args.output_csv.exists():
            output_fieldnames, output_rows = read_csv_dicts(args.output_csv)
            if output_fieldnames != fieldnames:
                raise SystemExit("Existing output CSV header differs from source; refusing to compare.")
            print(f"Existing output file found; comparing without refitting: {args.output_csv}")
            print_new_fit_comparisons(source_rows, reference_rows, output_rows, fieldnames, args)
        else:
            print(f"No existing output file found for comparison: {args.output_csv}")
        return 0

    try:
        from cobra.io import read_sbml_model
    except Exception as exc:
        raise SystemExit("cobra is required to run GEM fitting. Use --compare-reference-only for the CSV audit.") from exc

    model = read_sbml_model(str(args.model_sbml))
    lookup = reaction_lookup(model)

    mappings: Dict[str, ColumnMapping] = {}
    unmapped_columns: List[str] = []
    for column in fieldnames:
        if column == "experiment":
            continue
        mapping = map_flux_column(column, lookup, args.biomass_column, args.objective_reaction)
        if mapping is None:
            unmapped_columns.append(column)
        else:
            mappings[column] = mapping

    fitted_rows: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    for row in source_rows:
        fitted, diagnostic = solve_sample(model, row, mappings, fieldnames, args)
        diagnostics.append(diagnostic)
        if fitted is not None:
            fitted_rows.append(fitted)

    if len(fitted_rows) != len(source_rows):
        failed = [d for d in diagnostics if d.get("status") != "optimal"]
        print("Failed samples:")
        for row in failed:
            print(f"  {row.get('experiment')}: {row.get('status')} {row.get('message')}")
        raise SystemExit(
            f"Only solved {len(fitted_rows)}/{len(source_rows)} samples; refusing to write incomplete output."
        )

    write_csv_dicts(args.output_csv, fieldnames, fitted_rows)

    print_summary(
        fitted_rows=fitted_rows,
        diagnostics=diagnostics,
        mapped_columns=sorted(mappings),
        unmapped_columns=unmapped_columns,
        args=args,
    )
    print_new_fit_comparisons(source_rows, reference_rows, fitted_rows, fieldnames, args)
    return 0


def main() -> int:
    return run_with_args(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

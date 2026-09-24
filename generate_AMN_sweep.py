"""Deterministic AMN fructose/oxygen sweep; deploy with models/iML1515.xml.

Adapted from generate_ecoli_iML1515_AMN_data.py; no local helper imports.
Rows are fructose-major, oxygen-minor. Metadata sample_id is the zero-based
model CSV row number: fructose_index * oxygen_levels + oxygen_index.
Only complete grids are promoted from .partial.csv to final filenames.
"""

import argparse
import csv
import gc
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from cobra.flux_analysis import pfba
from cobra.io import read_sbml_model


CARBON_EXCHANGES = [
    # D-glucose is intentionally excluded to match the Faure-style AMN setup.
    "EX_rib__D_e",
    "EX_malt_e",
    "EX_melib_e",
    "EX_tre_e",
    "EX_fru_e",
    "EX_gal_e",
    "EX_ac_e",
    "EX_lac__D_e",
    "EX_succ_e",
    "EX_pyr_e",
]

FIXED_CARBON_EXCHANGES = [
    "EX_glyc_e",
]

BASE_EXCHANGES = [
    "EX_pi_e",
    "EX_co2_e",
    "EX_fe3_e",
    "EX_h_e",
    "EX_mn2_e",
    "EX_fe2_e",
    "EX_zn2_e",
    "EX_mg2_e",
    "EX_ca2_e",
    "EX_ni2_e",
    "EX_cu2_e",
    "EX_sel_e",
    "EX_cobalt2_e",
    "EX_h2o_e",
    "EX_mobd_e",
    "EX_so4_e",
    "EX_nh4_e",
    "EX_k_e",
    "EX_na1_e",
    "EX_cl_e",
    "EX_o2_e",
    "EX_tungs_e",
    "EX_slnt_e",
]

AMINO_EXCHANGES = [
    "EX_ala__L_e",
    "EX_pro__L_e",
    "EX_thr__L_e",
    "EX_gly_e",
]


def set_solver_timeout(model, timeout_seconds):
    if timeout_seconds is None or timeout_seconds <= 0:
        return
    try:
        model.solver.configuration.timeout = int(round(timeout_seconds))
    except Exception as exc:
        print(f"Warning: could not set solver timeout: {exc}")


def load_generation_model(model_dir, objective_reaction, solver_timeout_seconds):
    model = read_sbml_model(os.path.join(model_dir, "iML1515.xml"))
    model.objective = objective_reaction
    set_solver_timeout(model, solver_timeout_seconds)
    exchange_default_bounds = {
        rxn.id: (float(rxn.lower_bound), float(rxn.upper_bound))
        for rxn in model.exchanges
    }
    outputs = [rxn.id for rxn in model.reactions]
    return model, exchange_default_bounds, outputs


def validate_setup(model, input_cols, output_cols):
    counts = Counter(input_cols)
    duplicates = sorted(col for col, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError("Duplicate input columns: " + ", ".join(duplicates))

    missing_reactions = [ex for ex in input_cols if ex not in model.reactions]
    if missing_reactions:
        raise ValueError("Missing exchange reactions: " + ", ".join(missing_reactions))

    missing_tokens = [f"{ex}_flux" for ex in input_cols if f"{ex}_flux" not in output_cols]
    if missing_tokens:
        raise ValueError("Missing mapped output tokens: " + ", ".join(missing_tokens))


def reset_closed_medium(model, exchange_default_bounds):
    for rxn in model.exchanges:
        _, default_ub = exchange_default_bounds[rxn.id]
        rxn.upper_bound = max(0.0, float(default_ub))
        rxn.lower_bound = 0.0


def set_uptake(model, data, exchange_id, rate):
    model.reactions.get_by_id(exchange_id).lower_bound = -float(rate)
    data[exchange_id] = float(rate)


def solve_fluxes(model, flux_solver_mode, pfba_fraction_of_optimum):
    if flux_solver_mode == "pfba":
        return pfba(model, fraction_of_optimum=pfba_fraction_of_optimum)
    if flux_solver_mode == "fba":
        return model.optimize()
    raise ValueError("flux_solver_mode must be 'fba' or 'pfba'")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-prefix", default="iML1515_AMN_sweep")
    parser.add_argument("--carbon-levels", type=int, default=100)
    parser.add_argument("--oxygen-levels", type=int, default=100)
    parser.add_argument("--carbon-min", type=float, default=0.05)
    parser.add_argument("--carbon-max", type=float, default=2.2)
    parser.add_argument("--oxygen-min", type=float, default=1.0)
    parser.add_argument("--oxygen-max", type=float, default=10.0)
    parser.add_argument("--objective-reaction", default="BIOMASS_Ec_iML1515_core_75p37M")
    parser.add_argument("--flux-solver-mode", choices=["fba", "pfba"], default="fba")
    parser.add_argument("--pfba-fraction-of-optimum", type=float, default=0.999)
    parser.add_argument("--solver-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--solver-reset-interval", type=int, default=5000)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args(argv)


def validate_args(args):
    for name in ("carbon", "oxygen"):
        low, high = getattr(args, f"{name}_min"), getattr(args, f"{name}_max")
        if not (np.isfinite(low) and np.isfinite(high) and 0 <= low < high):
            raise ValueError(f"Require finite 0 <= {name}_min < {name}_max")
        if getattr(args, f"{name}_levels") < 2:
            raise ValueError(f"--{name}-levels must be >= 2")
    if not (0 < args.pfba_fraction_of_optimum <= 1):
        raise ValueError("--pfba-fraction-of-optimum must be in (0, 1]")
    if not np.isfinite(args.solver_timeout_seconds):
        raise ValueError("--solver-timeout-seconds must be finite")
    if args.solver_reset_interval < 0:
        raise ValueError("--solver-reset-interval must be >= 0 (0 disables reloads)")
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        raise ValueError("--output-prefix must be a filename prefix, not a path")


def generate_sweep_sample(model, exchange_default_bounds, outputs, args, carbon, oxygen):
    data = {}
    reset_closed_medium(model, exchange_default_bounds)
    for exchange in BASE_EXCHANGES:
        set_uptake(model, data, exchange, oxygen if exchange == "EX_o2_e" else 10.0)
    for exchange in FIXED_CARBON_EXCHANGES + AMINO_EXCHANGES:
        set_uptake(model, data, exchange, 2.2)
    for exchange in CARBON_EXCHANGES:
        set_uptake(model, data, exchange, carbon if exchange == "EX_fru_e" else 0.0)
    try:
        solution = solve_fluxes(model, args.flux_solver_mode, args.pfba_fraction_of_optimum)
        if solution.status != "optimal":
            return None, str(solution.status)
        fluxes = solution.fluxes.loc[outputs].to_numpy(dtype=float)
        if not np.isfinite(fluxes).all():
            return None, "nonfinite_fluxes"
    except Exception as exc:
        return None, f"error:{type(exc).__name__}: {exc}"
    data.update(zip((f"{rxn}_flux" for rxn in outputs), fluxes))
    return data, "optimal"


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    start = time.monotonic()
    planned = args.carbon_levels * args.oxygen_levels
    input_cols = BASE_EXCHANGES + FIXED_CARBON_EXCHANGES + AMINO_EXCHANGES + CARBON_EXCHANGES
    data_dir = Path(args.data_dir)
    final_paths = [data_dir / f"{args.output_prefix}_{planned}_{kind}.csv"
                   for kind in ("samples", "metadata")]
    partial_paths = [path.with_suffix(".partial.csv") for path in final_paths]
    for path in final_paths + partial_paths:
        if path.exists() and not args.overwrite_existing:
            raise FileExistsError(f"{path} exists; use --overwrite-existing or another prefix")

    model, bounds, outputs = load_generation_model(
        args.model_dir, args.objective_reaction, args.solver_timeout_seconds)
    output_cols = [f"{rxn}_flux" for rxn in outputs]
    validate_setup(model, input_cols, output_cols)
    for exchange in ("EX_fru_e", "EX_o2_e"):
        if exchange not in model.reactions:
            raise ValueError(f"Missing sweep exchange: {exchange}")
    ordered_columns = input_cols + output_cols
    carbon_levels = np.linspace(args.carbon_min, args.carbon_max, args.carbon_levels)
    oxygen_levels = np.linspace(args.oxygen_min, args.oxygen_max, args.oxygen_levels)
    data_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    print(f"Planned samples: {planned}; grid: {args.carbon_levels} x {args.oxygen_levels}")
    print(f"Objective: {args.objective_reaction}; solver mode: {args.flux_solver_mode}")
    print(f"pFBA fraction: {args.pfba_fraction_of_optimum}; timeout: {args.solver_timeout_seconds}s")
    print(f"Schema: {len(input_cols)} inputs + {len(outputs)} reaction fluxes")
    with partial_paths[0].open("w", newline="") as data_file, \
            partial_paths[1].open("w", newline="") as metadata_file:
        writer, metadata = csv.writer(data_file), csv.writer(metadata_file)
        writer.writerow(ordered_columns)
        metadata.writerow(["sample_id", "sweep_fructose_rate", "sweep_oxygen_rate",
                           "sweep_fructose_index", "sweep_oxygen_index"])
        for carbon_idx, carbon in enumerate(carbon_levels):
            for oxygen_idx, oxygen in enumerate(oxygen_levels):
                sample_id = carbon_idx * args.oxygen_levels + oxygen_idx
                if sample_id and args.solver_reset_interval and sample_id % args.solver_reset_interval == 0:
                    del model
                    gc.collect()
                    model, bounds, reloaded_outputs = load_generation_model(
                        args.model_dir, args.objective_reaction, args.solver_timeout_seconds)
                    if reloaded_outputs != outputs:
                        raise RuntimeError("Reaction order changed after model reload")
                    validate_setup(model, input_cols, output_cols)
                sample, status = generate_sweep_sample(
                    model, bounds, outputs, args, carbon, oxygen)
                counts[status] += 1
                if sample is None:
                    print(f"FAILED sample_id={sample_id}, fructose[{carbon_idx}]={carbon:.17g}, "
                          f"oxygen[{oxygen_idx}]={oxygen:.17g}: {status}", flush=True)
                else:
                    writer.writerow([sample[col] for col in ordered_columns])
                    metadata.writerow([sample_id, carbon, oxygen, carbon_idx, oxygen_idx])
                if (sample_id + 1) % 1000 == 0:
                    print(f"Evaluated {sample_id + 1}/{planned}; optimal={counts['optimal']}", flush=True)

    accepted = counts["optimal"]
    print(f"Total generated samples: {accepted}")
    print(f"Carbon range: [{args.carbon_min}, {args.carbon_max}]")
    print(f"Oxygen range: [{args.oxygen_min}, {args.oxygen_max}]")
    print(f"Grid dimensions: {args.carbon_levels} x {args.oxygen_levels}")
    print(f"Optimal: {accepted}; non-optimal/errors: {planned - accepted}; statuses: {dict(counts)}")
    print(f"Elapsed time: {time.monotonic() - start:.2f} seconds")
    if accepted != planned:
        raise RuntimeError(f"Incomplete sweep: {accepted}/{planned} optimal points. "
                           f"No final files written. Partial samples/metadata: {partial_paths}. "
                           "Metadata sample_id retains the original grid index; see FAILED lines above.")
    for partial, final in zip(partial_paths, final_paths):
        if args.overwrite_existing:
            os.replace(partial, final)
        else:
            os.rename(partial, final)
    print(f"Output filename: {final_paths[0]}")
    print(f"Metadata filename: {final_paths[1]}")


if __name__ == "__main__":
    main()



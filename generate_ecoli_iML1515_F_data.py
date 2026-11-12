"""Uncurated F sampler: choose 1--10 of all non-basal, non-oxygen exchanges."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import gc
import math
from pathlib import Path
import time

import numpy as np
from cobra.exceptions import OptimizationError
from cobra.flux_analysis import pfba

from generate_ecoli_iML1515_E_data import (
    DEFAULT_OBJECTIVE_REACTION,
    FIXED_BASE_EXCHANGES,
    MINN_CONTEXT_EXCHANGES,
    draw_uniform_subset,
    format_elapsed,
    load_generation_model,
    random_log_uniform_rate,
    reset_closed_medium,
    set_uptake,
    validate_range,
    validate_setup,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("training", "test"), default="training")
    parser.add_argument("--n-samples", type=int, help="Default training=1000000, test=50000.")
    parser.add_argument("--seed", type=int, help="Default training=42, test=9.")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-prefix", help="Default iML1515_F_{dataset}_data.")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--objective-reaction", default=DEFAULT_OBJECTIVE_REACTION)
    parser.add_argument("--pfba-fraction-of-optimum", type=float, default=0.999)
    parser.add_argument("--solver-timeout-seconds", type=int, default=120)
    parser.add_argument("--solver-reset-interval", type=int, default=5_000)
    parser.add_argument("--failure-reload-interval", type=int, default=100)
    parser.add_argument("--max-attempt-multiplier", type=float, default=20.0,
                        help="Maximum attempts per requested row; 0 disables the limit.")
    parser.add_argument("--base-rate", type=float, default=10.0)
    parser.add_argument("--source-rate-min", type=float, default=0.05)
    parser.add_argument("--source-rate-max", type=float, default=10.0)
    parser.add_argument("--oxygen-rate-min", type=float, default=1.0)
    parser.add_argument("--oxygen-rate-max", type=float, default=25.0)
    parser.add_argument("--max-sources", type=int, default=10,
                        help="Uniform source count from 1 through this value, inclusive.")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--progress-interval", type=int, default=1_000)
    parser.add_argument("--attempt-progress-interval", type=int, default=10_000)
    args = parser.parse_args(argv)
    if args.n_samples is None:
        args.n_samples = 1_000_000 if args.dataset == "training" else 50_000
    if args.seed is None:
        args.seed = 42 if args.dataset == "training" else 9
    if args.output_prefix is None:
        args.output_prefix = f"iML1515_F_{args.dataset}_data"
    return args


def validate_args(args):
    for name in ("n_samples", "batch_size", "max_sources"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("seed", "solver_timeout_seconds", "solver_reset_interval",
                 "failure_reload_interval", "progress_interval", "attempt_progress_interval"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("base_rate", "max_attempt_multiplier"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not 0 < args.pfba_fraction_of_optimum <= 1:
        raise ValueError("--pfba-fraction-of-optimum must be in (0, 1]")
    for name in ("source", "oxygen"):
        minimum = getattr(args, f"{name}_rate_min")
        maximum = getattr(args, f"{name}_rate_max")
        validate_range(f"F {name} uptake", minimum, maximum, strictly_positive=True)
        if round(minimum, 2) <= 0:
            raise ValueError(f"F {name} minimum must remain positive after rounding")
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        raise ValueError("--output-prefix must be a filename prefix, not a path")


def build_selectable_exchanges(model):
    """Preserve SBML exchange order; do not screen composition or growth support."""
    excluded = set(FIXED_BASE_EXCHANGES) | {"EX_o2_e"}
    return [reaction.id for reaction in model.exchanges if reaction.id not in excluded]


def build_input_columns(model):
    # Keep E's context/basal layout, followed by all remaining exchanges.
    leading = list(MINN_CONTEXT_EXCHANGES)
    leading += [exchange for exchange in FIXED_BASE_EXCHANGES if exchange not in leading]
    return leading + [exchange for exchange in build_selectable_exchanges(model)
                      if exchange not in leading]


def validate_generation_model(model, input_columns, output_columns, args):
    validate_setup(model, input_columns, output_columns)
    exchanges = list(model.exchanges)
    if set(input_columns) != {reaction.id for reaction in exchanges}:
        raise ValueError("F inputs must contain every model exchange exactly once")
    for reaction in exchanges:
        if list(reaction.metabolites.values()) != [-1.0]:
            raise ValueError(f"Expected negative uptake convention for {reaction.id}")
    pool_size = len(build_selectable_exchanges(model))
    if args.max_sources > pool_size:
        raise ValueError(f"--max-sources must not exceed the exchange pool size ({pool_size})")


def draw_medium(rng, input_columns, selectable_exchanges, args):
    data = dict.fromkeys(input_columns, 0.0)
    for exchange in FIXED_BASE_EXCHANGES:
        data[exchange] = args.base_rate
    data["EX_o2_e"] = random_log_uniform_rate(
        rng, args.oxygen_rate_min, args.oxygen_rate_max
    )
    source_count = int(rng.integers(1, args.max_sources + 1))
    for exchange in draw_uniform_subset(rng, selectable_exchanges, source_count):
        data[exchange] = random_log_uniform_rate(
            rng, args.source_rate_min, args.source_rate_max
        )
    return data


def generate_training_sample(model, rng, input_columns, outputs,
                             exchange_default_bounds, selectable_exchanges, args):
    data = draw_medium(rng, input_columns, selectable_exchanges, args)
    reset_closed_medium(model, exchange_default_bounds)
    for exchange, rate in data.items():
        if rate > 0:
            set_uptake(model, data, exchange, rate)
    try:
        solution = pfba(model, fraction_of_optimum=args.pfba_fraction_of_optimum)
    except OptimizationError:
        return None, f"solver_status:{model.solver.status}"
    if solution.status != "optimal":
        return None, f"solver_status:{solution.status}"
    fluxes = solution.fluxes.loc[outputs].to_numpy(dtype=float)
    if not np.isfinite(fluxes).all():
        return None, "nonfinite_flux"
    data.update(zip((f"{reaction}_flux" for reaction in outputs), fluxes))
    return data, "accepted"


def print_progress(sample_count, attempt_count, args, status_counts, start_time):
    elapsed = format_elapsed(time.time() - start_time)
    print(f"Generated {sample_count}/{args.n_samples} samples "
          f"(attempts={attempt_count}, feasible_rate={sample_count / max(1, attempt_count):.3f}, "
          f"elapsed={elapsed}, status_counts={dict(status_counts)})")


def run_generation(args):
    validate_args(args)
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    final_filename = data_dir / f"{args.output_prefix}_{args.n_samples}_samples.csv"
    if final_filename.exists() and not args.overwrite_existing:
        raise FileExistsError(f"Output already exists: {final_filename}. "
                              "Use --overwrite-existing to replace it.")

    model, exchange_default_bounds, outputs, model_path = load_generation_model(
        args.model_dir, args.objective_reaction, args.solver_timeout_seconds
    )
    selectable_exchanges = build_selectable_exchanges(model)
    input_columns = build_input_columns(model)
    output_columns = [f"{reaction}_flux" for reaction in outputs]
    ordered_columns = input_columns + output_columns
    validate_generation_model(model, input_columns, output_columns, args)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_filename = data_dir / f"{args.output_prefix}_temp_{run_stamp}.csv"
    max_attempts = None
    if args.max_attempt_multiplier > 0:
        max_attempts = max(args.n_samples, math.ceil(args.n_samples * args.max_attempt_multiplier))
    rng = np.random.default_rng(args.seed)
    print(f"Model: {model_path}")
    print(f"Solver: {model.solver.interface.__name__}")
    print(f"Arguments: {vars(args)}")
    print(f"Objective reaction: {args.objective_reaction}")
    print(f"Flux solver mode: pfba; fraction_of_optimum={args.pfba_fraction_of_optimum}")
    print(f"Dataset: {args.dataset}; random seed: {args.seed}")
    print(f"Input columns: {len(input_columns)}; output columns: {len(output_columns)}")
    print(f"Selectable exchanges: {len(selectable_exchanges)}; uniform count 1--{args.max_sources}")
    print(f"Fixed-base uptake cap: {args.base_rate:g}")
    print(f"Source uptake caps: log-uniform [{args.source_rate_min:g}, {args.source_rate_max:g}]")
    print(f"Oxygen uptake cap: log-uniform [{args.oxygen_rate_min:g}, {args.oxygen_rate_max:g}]")
    print("Uptake caps rounded to two decimals; optimal zero-biomass solutions are retained.")
    print(f"Temporary output: {temp_filename}")
    print(f"Planned final output: {final_filename}")

    start_time = time.time()
    sample_count = attempt_count = consecutive_failures = 0
    status_counts = Counter()
    batch = []
    with temp_filename.open("w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(ordered_columns)
        try:
            while sample_count < args.n_samples:
                if max_attempts is not None and attempt_count >= max_attempts:
                    raise RuntimeError(
                        f"Stopped after {attempt_count} attempts with only "
                        f"{sample_count}/{args.n_samples} accepted samples. "
                        f"Status counts: {dict(status_counts)}. Temporary output: {temp_filename}"
                    )
                periodic_reload = (args.solver_reset_interval > 0 and attempt_count > 0
                                   and attempt_count % args.solver_reset_interval == 0)
                failure_reload = (args.failure_reload_interval > 0
                                  and consecutive_failures >= args.failure_reload_interval)
                if periodic_reload or failure_reload:
                    del model
                    gc.collect()
                    model, exchange_default_bounds, reloaded_outputs, _ = load_generation_model(
                        args.model_dir, args.objective_reaction, args.solver_timeout_seconds
                    )
                    if reloaded_outputs != outputs or build_input_columns(model) != input_columns:
                        raise RuntimeError("Reaction order changed after reloading the model")
                    validate_generation_model(model, input_columns, output_columns, args)
                    consecutive_failures = 0

                attempt_count += 1
                sample, status = generate_training_sample(
                    model, rng, input_columns, outputs, exchange_default_bounds,
                    selectable_exchanges, args
                )
                status_counts[status] += 1
                if sample is not None:
                    batch.append([sample[column] for column in ordered_columns])
                    sample_count += 1
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                if len(batch) >= args.batch_size:
                    writer.writerows(batch)
                    file_handle.flush()
                    batch.clear()
                if ((args.progress_interval > 0 and sample is not None
                     and sample_count % args.progress_interval == 0)
                        or (args.attempt_progress_interval > 0
                            and attempt_count % args.attempt_progress_interval == 0)):
                    print_progress(sample_count, attempt_count, args, status_counts, start_time)
        finally:
            # Keep every accepted row available if an attempt limit or interruption stops the run.
            if batch:
                writer.writerows(batch)
            file_handle.flush()

    if final_filename.exists() and not args.overwrite_existing:
        raise FileExistsError(f"Output was created while this job was running: {final_filename}. "
                              f"Temporary output remains at {temp_filename}.")
    if args.overwrite_existing:
        temp_filename.replace(final_filename)
    else:
        temp_filename.rename(final_filename)
    print_progress(sample_count, attempt_count, args, status_counts, start_time)
    print(f"Saved to {final_filename}")


def main(argv=None):
    run_generation(parse_args(argv))


if __name__ == "__main__":
    main()

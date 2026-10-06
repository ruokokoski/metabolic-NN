import argparse
import csv
import gc
import math
import os
import time
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
from cobra.flux_analysis import pfba
from cobra.io import read_sbml_model

warnings.filterwarnings("ignore", message="Solver status is 'infeasible'")


CONTEXT_EXCHANGES = ["EX_glc__D_e", "EX_etoh_e", "EX_ac_e"]
BASE_EXCHANGES = [
    "EX_pi_e",
    "EX_co2_e",
    "EX_h_e",
    "EX_mn2_e",
    "EX_fe2_e",
    "EX_zn2_e",
    "EX_mg2_e",
    "EX_ca2_e",
    "EX_ni2_e",
    "EX_cu2_e",
    "EX_cobalt2_e",
    "EX_h2o_e",
    "EX_mobd_e",
    "EX_so4_e",
    "EX_nh4_e",
    "EX_k_e",
    "EX_na1_e",
    "EX_cl_e",
    "EX_o2_e",
    "EX_fe3_e",
    "EX_sel_e",
    "EX_tungs_e",
    "EX_slnt_e",
    "EX_cbl1_e",
]
SAMPLED_EXCHANGES = [
    "EX_glc__D_e",
    "EX_o2_e",
    "EX_co2_e",
    "EX_etoh_e",
    "EX_ac_e",
]
SECRETION_CONTEXT_EXCHANGES = ["EX_co2_e", "EX_etoh_e", "EX_ac_e"]
DEFAULT_OBJECTIVE_REACTION = "BIOMASS_Ec_iML1515_core_75p37M"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Tazza-style MINN training data from the E. coli "
            "iML1515 model."
        )
    )
    parser.add_argument("--n-samples", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--output-prefix", default="iML1515_MINN_tazza_training_data"
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace an existing final output file.",
    )
    parser.add_argument(
        "--objective-reaction", default=DEFAULT_OBJECTIVE_REACTION
    )
    parser.add_argument(
        "--flux-solver-mode", choices=("fba", "pfba"), default="pfba"
    )
    parser.add_argument(
        "--pfba-fraction-of-optimum", type=float, default=0.999
    )
    parser.add_argument("--solver-timeout-seconds", type=int, default=120)
    parser.add_argument("--solver-reset-interval", type=int, default=5_000)
    parser.add_argument("--failure-reload-interval", type=int, default=100)
    parser.add_argument("--max-attempt-multiplier", type=float, default=20.0)
    parser.add_argument("--base-rate", type=float, default=50.0)
    parser.add_argument("--glucose-cap-min", type=int, default=1)
    parser.add_argument("--glucose-cap-max", type=int, default=15)
    parser.add_argument("--oxygen-cap-min", type=int, default=1)
    parser.add_argument("--oxygen-cap-max", type=int, default=20)
    parser.add_argument("--co2-cap-min", type=int, default=0)
    parser.add_argument("--co2-cap-max", type=int, default=15)
    parser.add_argument("--ethanol-cap-min", type=int, default=0)
    parser.add_argument("--ethanol-cap-max", type=int, default=1)
    parser.add_argument("--acetate-cap-min", type=int, default=0)
    parser.add_argument("--acetate-cap-max", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--progress-interval", type=int, default=1_000)
    parser.add_argument("--attempt-progress-interval", type=int, default=10_000)
    return parser.parse_args()


def format_elapsed(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def random_integer_cap(rng, min_val, max_val):
    """Draw an integer cap uniformly from the inclusive range [min_val, max_val]."""
    if not isinstance(min_val, int) or not isinstance(max_val, int):
        raise TypeError("Integer cap endpoints must be integers.")
    if min_val < 0 or max_val < min_val:
        raise ValueError(f"Invalid integer cap range [{min_val}, {max_val}]")
    return int(rng.integers(min_val, max_val + 1))


def set_solver_timeout(model, timeout_seconds):
    if timeout_seconds <= 0:
        return

    configuration = getattr(model.solver, "configuration", None)
    if configuration is None or not hasattr(configuration, "timeout"):
        warnings.warn("The selected solver does not expose a timeout setting.")
        return

    configuration.timeout = int(timeout_seconds)


def load_generation_model(model_dir, objective_reaction, timeout_seconds):
    model_path = os.path.abspath(os.path.join(model_dir, "iML1515.xml"))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    model = read_sbml_model(model_path)
    if objective_reaction not in model.reactions:
        raise ValueError(
            f"Objective reaction {objective_reaction!r} is not present in {model_path}"
        )

    model.objective = objective_reaction
    set_solver_timeout(model, timeout_seconds)
    exchange_default_bounds = {
        reaction.id: (reaction.lower_bound, reaction.upper_bound)
        for reaction in model.exchanges
    }
    return model, exchange_default_bounds, model_path


def validate_setup(model, input_columns, sampled_cap_config):
    duplicates = sorted(
        column for column, count in Counter(input_columns).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate input columns: {duplicates}")

    required_reactions = set(input_columns) | set(sampled_cap_config)
    missing = sorted(
        reaction_id
        for reaction_id in required_reactions
        if reaction_id not in model.reactions
    )
    if missing:
        raise ValueError(f"Model is missing required reactions: {missing}")


def generate_training_sample(
    model,
    rng,
    context_exchanges,
    base_exchanges,
    outputs,
    base_rate,
    sampled_cap_config,
    exchange_default_bounds,
    flux_solver_mode,
    pfba_fraction_of_optimum,
):
    data = {}
    try:
        # Preserve the original Tazza medium semantics by restoring the model's
        # exchange defaults before applying the sampled and fixed caps.
        for reaction in model.exchanges:
            lower_bound, upper_bound = exchange_default_bounds[reaction.id]
            reaction.lower_bound = lower_bound
            reaction.upper_bound = upper_bound

        # Glucose is a sampled uptake cap. Ethanol and acetate are independent
        # sampled secretion caps. CSV cap values are stored as positive values.
        for exchange_id in context_exchanges:
            reaction = model.reactions.get_by_id(exchange_id)
            minimum, maximum = sampled_cap_config[exchange_id]
            rate = random_integer_cap(rng, minimum, maximum)

            if exchange_id == "EX_glc__D_e":
                reaction.lower_bound = -rate
            elif exchange_id in SECRETION_CONTEXT_EXCHANGES:
                reaction.lower_bound = 0.0
                reaction.upper_bound = rate
            else:
                raise ValueError(f"Unhandled context exchange: {exchange_id}")

            data[exchange_id] = rate

        # Oxygen is a sampled uptake cap and CO2 is a sampled secretion cap.
        # Other base exchanges retain fixed medium availability.
        for exchange_id in base_exchanges:
            reaction = model.reactions.get_by_id(exchange_id)
            if exchange_id == "EX_o2_e":
                minimum, maximum = sampled_cap_config[exchange_id]
                rate = random_integer_cap(rng, minimum, maximum)
                reaction.lower_bound = -rate
                data[exchange_id] = rate
            elif exchange_id in SECRETION_CONTEXT_EXCHANGES:
                minimum, maximum = sampled_cap_config[exchange_id]
                rate = random_integer_cap(rng, minimum, maximum)
                reaction.lower_bound = 0.0
                reaction.upper_bound = rate
                data[exchange_id] = rate
            else:
                reaction.lower_bound = -base_rate
                data[exchange_id] = base_rate

        if flux_solver_mode == "pfba":
            solution = pfba(
                model, fraction_of_optimum=pfba_fraction_of_optimum
            )
        else:
            solution = model.optimize()

        solution_status = getattr(solution, "status", "optimal")
        if solution_status != "optimal":
            return None, f"solver_status:{solution_status}"

        for reaction_id in outputs:
            data[f"{reaction_id}_flux"] = float(
                solution.fluxes.get(reaction_id, 0.0)
            )

        return data, "accepted"
    except Exception as exc:
        return None, f"error:{type(exc).__name__}"


def main():
    args = parse_args()

    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.base_rate < 0:
        raise ValueError("--base-rate must be non-negative")
    if not 0 < args.pfba_fraction_of_optimum <= 1:
        raise ValueError("--pfba-fraction-of-optimum must be in (0, 1]")

    sampled_cap_config = {
        "EX_glc__D_e": (args.glucose_cap_min, args.glucose_cap_max),
        "EX_o2_e": (args.oxygen_cap_min, args.oxygen_cap_max),
        "EX_co2_e": (args.co2_cap_min, args.co2_cap_max),
        "EX_etoh_e": (args.ethanol_cap_min, args.ethanol_cap_max),
        "EX_ac_e": (args.acetate_cap_min, args.acetate_cap_max),
    }
    for exchange_id, (minimum, maximum) in sampled_cap_config.items():
        if minimum < 0 or maximum < minimum:
            raise ValueError(
                f"Invalid cap range for {exchange_id}: [{minimum}, {maximum}]"
            )

    input_columns = CONTEXT_EXCHANGES + BASE_EXCHANGES
    rng = np.random.default_rng(args.seed)
    model, exchange_default_bounds, model_path = load_generation_model(
        args.model_dir, args.objective_reaction, args.solver_timeout_seconds
    )
    validate_setup(model, input_columns, sampled_cap_config)
    outputs = [reaction.id for reaction in model.reactions]
    output_columns = [f"{reaction_id}_flux" for reaction_id in outputs]
    ordered_columns = input_columns + output_columns

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    final_filename = os.path.join(
        data_dir, f"{args.output_prefix}_{args.n_samples}_samples.csv"
    )
    if os.path.exists(final_filename) and not args.overwrite_existing:
        raise FileExistsError(
            f"Output already exists: {final_filename}. "
            "Use --overwrite-existing to replace it."
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_filename = os.path.join(
        data_dir, f"{args.output_prefix}_temp_{run_stamp}.csv"
    )
    max_attempts = None
    if args.max_attempt_multiplier > 0:
        max_attempts = max(
            args.n_samples,
            math.ceil(args.n_samples * args.max_attempt_multiplier),
        )

    print(f"Model: {model_path}")
    print(f"Objective reaction: {args.objective_reaction}")
    print(f"Flux solver mode: {args.flux_solver_mode}")
    if args.flux_solver_mode == "pfba":
        print(f"pFBA fraction_of_optimum: {args.pfba_fraction_of_optimum}")
    print(f"Generating {args.n_samples} training samples")
    print("Sampled integer reservoir cap configuration (inclusive min, max):")
    for exchange_id in SAMPLED_EXCHANGES:
        print(f"  {exchange_id}: {sampled_cap_config[exchange_id]}")
    print(
        "Glucose and oxygen are uptake caps; CO2, ethanol, and acetate "
        "are secretion caps."
    )
    print(f"Temporary output: {temp_filename}")

    start_time = time.time()
    sample_count = 0
    attempt_count = 0
    consecutive_failures = 0
    status_counts = Counter()
    batch = []

    with open(temp_filename, "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(ordered_columns)

        while sample_count < args.n_samples:
            if max_attempts is not None and attempt_count >= max_attempts:
                raise RuntimeError(
                    f"Stopped after {attempt_count} attempts with only "
                    f"{sample_count}/{args.n_samples} accepted samples. "
                    f"Failure counts: {dict(status_counts)}. Temporary output: "
                    f"{temp_filename}"
                )

            if (
                args.solver_reset_interval > 0
                and attempt_count > 0
                and attempt_count % args.solver_reset_interval == 0
            ):
                del model
                gc.collect()
                model, exchange_default_bounds, _ = load_generation_model(
                    args.model_dir,
                    args.objective_reaction,
                    args.solver_timeout_seconds,
                )
                validate_setup(model, input_columns, sampled_cap_config)

            attempt_count += 1
            sample, status = generate_training_sample(
                model=model,
                rng=rng,
                context_exchanges=CONTEXT_EXCHANGES,
                base_exchanges=BASE_EXCHANGES,
                outputs=outputs,
                base_rate=args.base_rate,
                sampled_cap_config=sampled_cap_config,
                exchange_default_bounds=exchange_default_bounds,
                flux_solver_mode=args.flux_solver_mode,
                pfba_fraction_of_optimum=args.pfba_fraction_of_optimum,
            )
            status_counts[status] += 1

            if sample is not None:
                batch.append([sample.get(column, 0.0) for column in ordered_columns])
                sample_count += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            if len(batch) >= args.batch_size:
                writer.writerows(batch)
                file_handle.flush()
                batch = []

            if (
                args.progress_interval > 0
                and sample_count > 0
                and sample_count % args.progress_interval == 0
                and status == "accepted"
            ):
                elapsed = time.time() - start_time
                feasible_rate = sample_count / attempt_count
                print(
                    f"Generated {sample_count}/{args.n_samples} samples "
                    f"(attempts={attempt_count}, feasible_rate={feasible_rate:.3f}, "
                    f"elapsed={format_elapsed(elapsed)}, "
                    f"time={datetime.now().strftime('%H:%M')})"
                )
            elif (
                args.attempt_progress_interval > 0
                and attempt_count % args.attempt_progress_interval == 0
            ):
                print(
                    f"Attempts={attempt_count}, accepted={sample_count}, "
                    f"failure_counts={dict(status_counts)}"
                )

            if (
                args.failure_reload_interval > 0
                and consecutive_failures >= args.failure_reload_interval
            ):
                print(
                    f"Reloading model after {consecutive_failures} consecutive "
                    "failed attempts"
                )
                del model
                gc.collect()
                model, exchange_default_bounds, _ = load_generation_model(
                    args.model_dir,
                    args.objective_reaction,
                    args.solver_timeout_seconds,
                )
                validate_setup(model, input_columns, sampled_cap_config)
                consecutive_failures = 0

        if batch:
            writer.writerows(batch)
            file_handle.flush()

    if os.path.exists(final_filename):
        if not args.overwrite_existing:
            raise FileExistsError(
                f"Output was created while this job was running: {final_filename}. "
                f"Temporary output remains at {temp_filename}."
            )
        os.replace(temp_filename, final_filename)
    else:
        os.rename(temp_filename, final_filename)

    total_time = time.time() - start_time
    print(f"Completed {sample_count} samples in {format_elapsed(total_time)}")
    print(f"Attempts: {attempt_count}")
    print(f"Status counts: {dict(status_counts)}")
    print(f"Saved to {final_filename}")


if __name__ == "__main__":
    main()

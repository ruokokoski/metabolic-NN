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


MINN_CONTEXT_EXCHANGES = [
    "EX_glc__D_e",
    "EX_o2_e",
    "EX_co2_e",
    "EX_etoh_e",
    "EX_ac_e",
]

COMMON_BASE_EXCHANGES = [
    "EX_pi_e",
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
    "EX_tungs_e",
    "EX_slnt_e",
]

B_ONLY_BASE_EXCHANGES = [
    "EX_cbl1_e",
]

AMN_CARBON_EXCHANGES = [
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

NON_CONTEXT_AMN_CARBON_EXCHANGES = [
    exchange_id
    for exchange_id in AMN_CARBON_EXCHANGES
    if exchange_id not in set(MINN_CONTEXT_EXCHANGES)
]

FIXED_CARBON_EXCHANGES = [
    "EX_glyc_e",
]

AMINO_EXCHANGES = [
    "EX_ala__L_e",
    "EX_pro__L_e",
    "EX_thr__L_e",
    "EX_gly_e",
]

DEFAULT_OBJECTIVE_REACTION = "BIOMASS_Ec_iML1515_core_75p37M"
A_REGIME = 0
B_REGIME = 1
REGIME_NAMES = {
    A_REGIME: "a",
    B_REGIME: "b",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced union of AMN-style A and Tazza-style MINN B "
            "iML1515 samples using pFBA."
        )
    )
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--output-prefix", default="iML1515_AB_union_training_data"
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
        "--pfba-fraction-of-optimum", type=float, default=0.999
    )
    parser.add_argument("--solver-timeout-seconds", type=int, default=120)
    parser.add_argument("--solver-reset-interval", type=int, default=5_000)
    parser.add_argument("--failure-reload-interval", type=int, default=100)
    parser.add_argument("--max-attempt-multiplier", type=float, default=20.0)

    parser.add_argument("--a-base-rate", type=float, default=10.0)
    parser.add_argument("--a-carbon-rate-min", type=float, default=0.05)
    parser.add_argument("--a-carbon-rate-max", type=float, default=2.2)
    parser.add_argument("--a-fixed-carbon-rate", type=float, default=2.2)
    parser.add_argument("--a-amino-rate", type=float, default=2.2)
    parser.add_argument("--a-oxygen-rate-min", type=float, default=1.0)
    parser.add_argument("--a-oxygen-rate-max", type=float, default=10.0)
    parser.add_argument("--a-max-carbon-sources", type=int, default=4)

    parser.add_argument("--b-base-rate", type=float, default=50.0)
    parser.add_argument("--b-glucose-cap-min", type=int, default=1)
    parser.add_argument("--b-glucose-cap-max", type=int, default=15)
    parser.add_argument("--b-oxygen-cap-min", type=int, default=1)
    parser.add_argument("--b-oxygen-cap-max", type=int, default=20)
    parser.add_argument("--b-co2-cap-min", type=int, default=0)
    parser.add_argument("--b-co2-cap-max", type=int, default=15)
    parser.add_argument("--b-ethanol-cap-min", type=int, default=0)
    parser.add_argument("--b-ethanol-cap-max", type=int, default=1)
    parser.add_argument("--b-acetate-cap-min", type=int, default=0)
    parser.add_argument("--b-acetate-cap-max", type=int, default=3)

    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--progress-interval", type=int, default=1_000)
    parser.add_argument("--attempt-progress-interval", type=int, default=10_000)
    return parser.parse_args()


def format_elapsed(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def random_rate(rng, min_val, max_val):
    return round(float(rng.uniform(min_val, max_val)), 2)


def random_integer_cap(rng, min_val, max_val):
    return int(rng.integers(min_val, max_val + 1))


def draw_amn_carbon_subset(rng, max_sources):
    count = int(rng.integers(1, min(max_sources, len(AMN_CARBON_EXCHANGES)) + 1))
    return rng.choice(
        AMN_CARBON_EXCHANGES,
        size=count,
        replace=False,
    ).tolist()


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
        reaction.id: (
            float(reaction.lower_bound),
            float(reaction.upper_bound),
        )
        for reaction in model.exchanges
    }
    outputs = [reaction.id for reaction in model.reactions]
    return model, exchange_default_bounds, outputs, model_path


def build_input_columns():
    return (
        list(MINN_CONTEXT_EXCHANGES)
        + list(COMMON_BASE_EXCHANGES)
        + list(B_ONLY_BASE_EXCHANGES)
        + list(FIXED_CARBON_EXCHANGES)
        + list(AMINO_EXCHANGES)
        + list(NON_CONTEXT_AMN_CARBON_EXCHANGES)
    )


def validate_setup(model, input_columns, output_columns):
    duplicates = sorted(
        column for column, count in Counter(input_columns).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate input columns: {duplicates}")

    missing_reactions = sorted(
        reaction_id
        for reaction_id in input_columns
        if reaction_id not in model.reactions
    )
    if missing_reactions:
        raise ValueError(f"Model is missing required reactions: {missing_reactions}")

    missing_tokens = sorted(
        f"{reaction_id}_flux"
        for reaction_id in input_columns
        if f"{reaction_id}_flux" not in output_columns
    )
    if missing_tokens:
        raise ValueError(f"Missing mapped output tokens: {missing_tokens}")


def validate_args(args):
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.n_samples % 2 != 0:
        raise ValueError("--n-samples must be even for an exact 50/50 A/B union")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0 < args.pfba_fraction_of_optimum <= 1:
        raise ValueError("--pfba-fraction-of-optimum must be in (0, 1]")
    if not 1 <= args.a_max_carbon_sources <= len(AMN_CARBON_EXCHANGES):
        raise ValueError(
            "--a-max-carbon-sources must be between 1 and "
            f"{len(AMN_CARBON_EXCHANGES)}"
        )

    nonnegative_values = {
        "A base rate": args.a_base_rate,
        "A fixed-carbon rate": args.a_fixed_carbon_rate,
        "A amino-acid rate": args.a_amino_rate,
        "B base rate": args.b_base_rate,
    }
    for name, value in nonnegative_values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    continuous_ranges = {
        "A carbon rate": (args.a_carbon_rate_min, args.a_carbon_rate_max),
        "A oxygen rate": (args.a_oxygen_rate_min, args.a_oxygen_rate_max),
    }
    for name, (minimum, maximum) in continuous_ranges.items():
        if minimum < 0 or maximum < minimum:
            raise ValueError(f"Invalid {name} range [{minimum}, {maximum}]")

    integer_ranges = build_b_cap_config(args)
    for exchange_id, (minimum, maximum) in integer_ranges.items():
        if minimum < 0 or maximum < minimum:
            raise ValueError(
                f"Invalid B cap range for {exchange_id}: [{minimum}, {maximum}]"
            )


def build_b_cap_config(args):
    return {
        "EX_glc__D_e": (args.b_glucose_cap_min, args.b_glucose_cap_max),
        "EX_o2_e": (args.b_oxygen_cap_min, args.b_oxygen_cap_max),
        "EX_co2_e": (args.b_co2_cap_min, args.b_co2_cap_max),
        "EX_etoh_e": (args.b_ethanol_cap_min, args.b_ethanol_cap_max),
        "EX_ac_e": (args.b_acetate_cap_min, args.b_acetate_cap_max),
    }


def build_regime_schedule(n_samples, rng):
    samples_per_regime = n_samples // 2
    schedule = np.empty(n_samples, dtype=np.uint8)
    schedule[:samples_per_regime] = A_REGIME
    schedule[samples_per_regime:] = B_REGIME
    rng.shuffle(schedule)
    return schedule


def reset_closed_medium(model, exchange_default_bounds):
    for reaction in model.exchanges:
        _, default_upper_bound = exchange_default_bounds[reaction.id]
        reaction.lower_bound = 0.0
        reaction.upper_bound = max(0.0, default_upper_bound)


def restore_default_medium(model, exchange_default_bounds):
    for reaction in model.exchanges:
        lower_bound, upper_bound = exchange_default_bounds[reaction.id]
        reaction.lower_bound = lower_bound
        reaction.upper_bound = upper_bound


def set_uptake(model, data, exchange_id, rate):
    model.reactions.get_by_id(exchange_id).lower_bound = -float(rate)
    data[exchange_id] = rate


def set_secretion_cap(model, data, exchange_id, cap):
    reaction = model.reactions.get_by_id(exchange_id)
    reaction.lower_bound = 0.0
    reaction.upper_bound = float(cap)
    data[exchange_id] = int(cap)


def apply_a_regime(model, data, rng, exchange_default_bounds, args):
    reset_closed_medium(model, exchange_default_bounds)

    for exchange_id in COMMON_BASE_EXCHANGES:
        set_uptake(model, data, exchange_id, args.a_base_rate)
    set_uptake(model, data, "EX_co2_e", args.a_base_rate)

    oxygen_rate = random_rate(
        rng,
        args.a_oxygen_rate_min,
        args.a_oxygen_rate_max,
    )
    set_uptake(model, data, "EX_o2_e", oxygen_rate)

    for exchange_id in draw_amn_carbon_subset(rng, args.a_max_carbon_sources):
        carbon_rate = random_rate(
            rng,
            args.a_carbon_rate_min,
            args.a_carbon_rate_max,
        )
        set_uptake(model, data, exchange_id, carbon_rate)

    for exchange_id in FIXED_CARBON_EXCHANGES:
        set_uptake(model, data, exchange_id, args.a_fixed_carbon_rate)
    for exchange_id in AMINO_EXCHANGES:
        set_uptake(model, data, exchange_id, args.a_amino_rate)


def apply_b_regime(model, data, rng, exchange_default_bounds, b_cap_config, args):
    restore_default_medium(model, exchange_default_bounds)

    for exchange_id in COMMON_BASE_EXCHANGES + B_ONLY_BASE_EXCHANGES:
        set_uptake(model, data, exchange_id, args.b_base_rate)

    glucose_cap = random_integer_cap(rng, *b_cap_config["EX_glc__D_e"])
    oxygen_cap = random_integer_cap(rng, *b_cap_config["EX_o2_e"])
    set_uptake(model, data, "EX_glc__D_e", glucose_cap)
    set_uptake(model, data, "EX_o2_e", oxygen_cap)

    for exchange_id in ("EX_co2_e", "EX_etoh_e", "EX_ac_e"):
        secretion_cap = random_integer_cap(rng, *b_cap_config[exchange_id])
        set_secretion_cap(model, data, exchange_id, secretion_cap)


def generate_training_sample(
    model,
    rng,
    regime,
    input_columns,
    outputs,
    exchange_default_bounds,
    b_cap_config,
    args,
):
    data = {column: 0.0 for column in input_columns}

    try:
        if regime == "a":
            apply_a_regime(model, data, rng, exchange_default_bounds, args)
        elif regime == "b":
            apply_b_regime(
                model,
                data,
                rng,
                exchange_default_bounds,
                b_cap_config,
                args,
            )
        else:
            raise ValueError(f"Unknown regime: {regime}")

        solution = pfba(
            model,
            fraction_of_optimum=args.pfba_fraction_of_optimum,
        )
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


def print_progress(
    sample_count,
    attempt_count,
    n_samples,
    accepted_regime_counts,
    status_counts,
    start_time,
):
    elapsed = time.time() - start_time
    feasible_rate = sample_count / max(1, attempt_count)
    regime_summary = ", ".join(
        f"{regime}={accepted_regime_counts[regime]}" for regime in ("a", "b")
    )
    print(
        f"Generated {sample_count}/{n_samples} samples "
        f"(attempts={attempt_count}, feasible_rate={feasible_rate:.3f}, "
        f"accepted regimes: {regime_summary}, elapsed={format_elapsed(elapsed)}, "
        f"time={datetime.now().strftime('%H:%M')}, "
        f"status_counts={dict(status_counts)})"
    )


def main():
    args = parse_args()
    validate_args(args)

    input_columns = build_input_columns()
    b_cap_config = build_b_cap_config(args)

    seed_sequence = np.random.SeedSequence(args.seed)
    schedule_seed, sample_seed = seed_sequence.spawn(2)
    schedule_rng = np.random.default_rng(schedule_seed)
    sample_rng = np.random.default_rng(sample_seed)
    regime_schedule = build_regime_schedule(args.n_samples, schedule_rng)

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    final_filename = os.path.join(
        data_dir,
        f"{args.output_prefix}_{args.n_samples}_samples.csv",
    )
    if os.path.exists(final_filename) and not args.overwrite_existing:
        raise FileExistsError(
            f"Output already exists: {final_filename}. "
            "Use --overwrite-existing to replace it."
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_filename = os.path.join(
        data_dir,
        f"{args.output_prefix}_temp_{run_stamp}.csv",
    )

    model, exchange_default_bounds, outputs, model_path = load_generation_model(
        args.model_dir,
        args.objective_reaction,
        args.solver_timeout_seconds,
    )
    output_columns = [f"{reaction_id}_flux" for reaction_id in outputs]
    ordered_columns = input_columns + output_columns
    validate_setup(model, input_columns, output_columns)

    max_attempts = None
    if args.max_attempt_multiplier > 0:
        max_attempts = max(
            args.n_samples,
            math.ceil(args.n_samples * args.max_attempt_multiplier),
        )

    print(f"Model: {model_path}")
    print(f"Objective reaction: {args.objective_reaction}")
    print("Flux solver mode: pfba")
    print(f"pFBA fraction_of_optimum: {args.pfba_fraction_of_optimum}")
    print(f"Random seed: {args.seed}")
    print(f"Input columns: {len(input_columns)}")
    print(f"Output columns: {len(output_columns)}")
    print(
        f"Balanced accepted-sample targets: "
        f"A={args.n_samples // 2}, B={args.n_samples // 2}"
    )
    print(
        "A oxygen uptake range: "
        f"continuous [{args.a_oxygen_rate_min:g}, {args.a_oxygen_rate_max:g}]"
    )
    print(
        "B oxygen uptake range: "
        f"integer [{args.b_oxygen_cap_min}, {args.b_oxygen_cap_max}]"
    )
    print("B sampled integer bounds (inclusive min, max):")
    for exchange_id in MINN_CONTEXT_EXCHANGES:
        print(f"  {exchange_id}: {b_cap_config[exchange_id]}")
    print(f"Temporary output: {temp_filename}")
    print(f"Planned final output: {final_filename}")

    start_time = time.time()
    sample_count = 0
    attempt_count = 0
    consecutive_failures = 0
    accepted_regime_counts = Counter()
    attempted_regime_counts = Counter()
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
                    f"Accepted regimes: {dict(accepted_regime_counts)}. "
                    f"Failure counts: {dict(status_counts)}. "
                    f"Temporary output: {temp_filename}"
                )

            if (
                args.solver_reset_interval > 0
                and attempt_count > 0
                and attempt_count % args.solver_reset_interval == 0
            ):
                del model
                gc.collect()
                model, exchange_default_bounds, reloaded_outputs, _ = (
                    load_generation_model(
                        args.model_dir,
                        args.objective_reaction,
                        args.solver_timeout_seconds,
                    )
                )
                if reloaded_outputs != outputs:
                    raise RuntimeError("Reaction order changed after reloading the model")
                validate_setup(model, input_columns, output_columns)

            regime = REGIME_NAMES[int(regime_schedule[sample_count])]
            attempt_count += 1
            attempted_regime_counts[regime] += 1
            sample, status = generate_training_sample(
                model=model,
                rng=sample_rng,
                regime=regime,
                input_columns=input_columns,
                outputs=outputs,
                exchange_default_bounds=exchange_default_bounds,
                b_cap_config=b_cap_config,
                args=args,
            )
            status_counts[f"{regime}:{status}"] += 1

            if sample is not None:
                batch.append(
                    [sample.get(column, 0.0) for column in ordered_columns]
                )
                sample_count += 1
                accepted_regime_counts[regime] += 1
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
                print_progress(
                    sample_count=sample_count,
                    attempt_count=attempt_count,
                    n_samples=args.n_samples,
                    accepted_regime_counts=accepted_regime_counts,
                    status_counts=status_counts,
                    start_time=start_time,
                )
            elif (
                args.attempt_progress_interval > 0
                and attempt_count % args.attempt_progress_interval == 0
            ):
                print_progress(
                    sample_count=sample_count,
                    attempt_count=attempt_count,
                    n_samples=args.n_samples,
                    accepted_regime_counts=accepted_regime_counts,
                    status_counts=status_counts,
                    start_time=start_time,
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
                model, exchange_default_bounds, reloaded_outputs, _ = (
                    load_generation_model(
                        args.model_dir,
                        args.objective_reaction,
                        args.solver_timeout_seconds,
                    )
                )
                if reloaded_outputs != outputs:
                    raise RuntimeError("Reaction order changed after reloading the model")
                validate_setup(model, input_columns, output_columns)
                consecutive_failures = 0

        if batch:
            writer.writerows(batch)
            file_handle.flush()

    expected_per_regime = args.n_samples // 2
    if any(
        accepted_regime_counts[regime] != expected_per_regime
        for regime in ("a", "b")
    ):
        raise RuntimeError(
            "Accepted regime counts do not match the requested balanced union: "
            f"{dict(accepted_regime_counts)}"
        )

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
    print(f"Attempted regime counts: {dict(attempted_regime_counts)}")
    print(f"Accepted regime counts: {dict(accepted_regime_counts)}")
    print(f"Status counts: {dict(status_counts)}")
    print(f"Saved to {final_filename}")


if __name__ == "__main__":
    main()

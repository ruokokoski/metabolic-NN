import argparse
import csv
import gc
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

MINN_SECRETION_CONTEXT_EXCHANGES = [
    "EX_co2_e",
    "EX_etoh_e",
    "EX_ac_e",
]

FAURE_CARBON_EXCHANGES = [
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

NON_CONTEXT_FAURE_CARBON_EXCHANGES = [
    ex for ex in FAURE_CARBON_EXCHANGES if ex not in set(MINN_CONTEXT_EXCHANGES)
]

FIXED_CARBON_EXCHANGES = [
    "EX_glyc_e",
]

BASE_EXCHANGES = [
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

AMINO_EXCHANGES = [
    "EX_ala__L_e",
    "EX_pro__L_e",
    "EX_thr__L_e",
    "EX_gly_e",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate shared AMN/MINN iML1515 FluxTransformer samples. "
            "This keeps Faure-like no-glucose media while adding glucose and "
            "MINN's five reservoir-context exchanges."
        )
    )
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-prefix", default="iML1515_AMN_MINN_test_data")
    parser.add_argument("--overwrite-existing", action="store_true")

    parser.add_argument("--objective-reaction", default="BIOMASS_Ec_iML1515_core_75p37M")
    parser.add_argument("--flux-solver-mode", choices=["fba", "pfba"], default="pfba")
    parser.add_argument("--pfba-fraction-of-optimum", type=float, default=0.999)
    parser.add_argument("--solver-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--solver-reset-interval", type=int, default=5000)
    parser.add_argument("--failure-reload-interval", type=int, default=100)
    parser.add_argument("--max-attempt-multiplier", type=float, default=20.0)

    parser.add_argument("--regime-minn-weight", type=float, default=0.50)
    parser.add_argument("--regime-faure-weight", type=float, default=0.40)
    parser.add_argument("--regime-mixed-weight", type=float, default=0.10)

    parser.add_argument("--base-rate", type=float, default=50.0)
    parser.add_argument("--glucose-rate-min", type=int, default=1)
    parser.add_argument("--glucose-rate-max", type=int, default=15)
    parser.add_argument("--oxygen-rate-min", type=int, default=1)
    parser.add_argument("--oxygen-rate-max", type=int, default=20)
    parser.add_argument("--co2-secretion-cap-min", type=int, default=0)
    parser.add_argument("--co2-secretion-cap-max", type=int, default=15)
    parser.add_argument("--ethanol-secretion-cap-min", type=int, default=0)
    parser.add_argument("--ethanol-secretion-cap-max", type=int, default=1)
    parser.add_argument("--acetate-secretion-cap-min", type=int, default=0)
    parser.add_argument("--acetate-secretion-cap-max", type=int, default=3)
    parser.add_argument("--faure-oxygen-rate-min", type=float, default=1.0)
    parser.add_argument("--faure-oxygen-rate-max", type=float, default=10.0)
    parser.add_argument("--carbon-rate-min", type=float, default=0.05)
    parser.add_argument("--carbon-rate-max", type=float, default=2.2)
    parser.add_argument("--fixed-carbon-rate", type=float, default=2.2)
    parser.add_argument("--amino-rate", type=float, default=2.2)
    parser.add_argument("--max-faure-carbon-sources", type=int, default=4)
    parser.add_argument("--max-mixed-extra-carbon-sources", type=int, default=2)
    parser.add_argument("--mixed-supplement-probability", type=float, default=0.0)

    parser.add_argument(
        "--exclude-cbl1",
        action="store_true",
        help=(
            "Exclude the fixed cobalamin exchange EX_cbl1_e from the base "
            "medium. By default EX_cbl1_e is included because it is required "
            "for the wild-type objective BIOMASS_Ec_iML1515_WT_75p37M and is "
            "harmless for the core objective."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--attempt-progress-interval", type=int, default=10000)
    return parser.parse_args()


def random_rate(rng, min_val, max_val, log_uniform=False):
    if min_val < 0 or max_val < 0 or max_val < min_val:
        raise ValueError(f"Invalid rate range [{min_val}, {max_val}]")
    if log_uniform:
        if min_val <= 0:
            raise ValueError("Log-uniform sampling requires min_val > 0")
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        return round(float(10 ** rng.uniform(log_min, log_max)), 2)
    return round(float(rng.uniform(min_val, max_val)), 2)


def random_integer_cap(rng, min_val, max_val):
    """Draw an integer cap uniformly from the inclusive range [min_val, max_val]."""
    if not isinstance(min_val, int) or not isinstance(max_val, int):
        raise TypeError("Integer cap endpoints must be integers.")
    if min_val < 0 or max_val < min_val:
        raise ValueError(f"Invalid integer cap range [{min_val}, {max_val}]")
    return int(rng.integers(min_val, max_val + 1))


def draw_subset(rng, exchanges, max_sources, min_sources=0):
    max_sources = min(max_sources, len(exchanges))
    min_sources = min(min_sources, max_sources)
    k = rng.integers(min_sources, max_sources + 1)
    if k <= 0:
        return []
    return rng.choice(exchanges, size=int(k), replace=False).tolist()


def format_elapsed(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}h {minutes}m"


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


def build_input_cols(exclude_cbl1=False):
    base_exchanges = list(BASE_EXCHANGES)
    if not exclude_cbl1:
        base_exchanges.append("EX_cbl1_e")

    input_cols = (
        list(MINN_CONTEXT_EXCHANGES)
        + base_exchanges
        + FIXED_CARBON_EXCHANGES
        + AMINO_EXCHANGES
        + NON_CONTEXT_FAURE_CARBON_EXCHANGES
    )
    return input_cols, base_exchanges


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
    # Preserve integer-valued MINN caps in the CSV while still accepting
    # continuous Faure carbon/oxygen rates.
    data[exchange_id] = rate


def set_secretion_only(model, exchange_default_bounds, exchange_id):
    rxn = model.reactions.get_by_id(exchange_id)
    _, default_ub = exchange_default_bounds[exchange_id]
    rxn.lower_bound = 0.0
    rxn.upper_bound = max(0.0, float(default_ub))


def set_secretion_cap(model, data, exchange_id, cap):
    rxn = model.reactions.get_by_id(exchange_id)
    rxn.lower_bound = 0.0
    rxn.upper_bound = float(cap)
    data[exchange_id] = int(cap)


def solve_fluxes(model, flux_solver_mode, pfba_fraction_of_optimum):
    if flux_solver_mode == "pfba":
        return pfba(model, fraction_of_optimum=pfba_fraction_of_optimum)
    if flux_solver_mode == "fba":
        return model.optimize()
    raise ValueError("flux_solver_mode must be 'fba' or 'pfba'")


def normalize_regime_weights(args):
    weights = {
        "minn": float(args.regime_minn_weight),
        "faure": float(args.regime_faure_weight),
        "mixed": float(args.regime_mixed_weight),
    }
    negative = {k: v for k, v in weights.items() if v < 0}
    if negative:
        raise ValueError(f"Regime weights must be nonnegative: {negative}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one regime weight must be positive.")
    names = list(weights)
    probabilities = np.array([weights[name] / total for name in names], dtype=np.float64)
    return names, probabilities


def choose_regime(rng, regime_names, regime_probabilities):
    idx = int(rng.choice(len(regime_names), p=regime_probabilities))
    return regime_names[idx]


def apply_base_medium(model, data, base_exchanges, rate):
    for ex in base_exchanges:
        set_uptake(model, data, ex, rate)


def apply_minn_context_caps(model, data, rng, args):
    glc_rate = random_integer_cap(
        rng,
        args.glucose_rate_min,
        args.glucose_rate_max,
    )
    o2_rate = random_integer_cap(
        rng,
        args.oxygen_rate_min,
        args.oxygen_rate_max,
    )
    secretion_caps = {
        "EX_co2_e": random_integer_cap(
            rng,
            args.co2_secretion_cap_min,
            args.co2_secretion_cap_max,
        ),
        "EX_etoh_e": random_integer_cap(
            rng,
            args.ethanol_secretion_cap_min,
            args.ethanol_secretion_cap_max,
        ),
        "EX_ac_e": random_integer_cap(
            rng,
            args.acetate_secretion_cap_min,
            args.acetate_secretion_cap_max,
        ),
    }

    set_uptake(model, data, "EX_glc__D_e", glc_rate)
    set_uptake(model, data, "EX_o2_e", o2_rate)
    for ex, cap in secretion_caps.items():
        set_secretion_cap(model, data, ex, cap)


def apply_minn_regime(model, data, rng, base_exchanges, args):
    apply_base_medium(model, data, base_exchanges, args.base_rate)
    apply_minn_context_caps(model, data, rng, args)


def apply_faure_regime(model, data, rng, exchange_default_bounds, base_exchanges, args):
    apply_base_medium(model, data, base_exchanges, args.base_rate)
    set_uptake(model, data, "EX_co2_e", args.base_rate)

    # Keep the Faure AMN medium sampling unchanged except for oxygen: Faure
    # holds oxygen fixed, whereas this FluxTransformer generator samples it.
    o2_rate = random_rate(
        rng,
        args.faure_oxygen_rate_min,
        args.faure_oxygen_rate_max,
    )
    set_uptake(model, data, "EX_o2_e", o2_rate)

    carbon_subset = draw_subset(
        rng,
        FAURE_CARBON_EXCHANGES,
        max_sources=args.max_faure_carbon_sources,
        min_sources=1,
    )
    for ex in carbon_subset:
        rate = random_rate(rng, args.carbon_rate_min, args.carbon_rate_max)
        set_uptake(model, data, ex, rate)

    for ex in FIXED_CARBON_EXCHANGES:
        set_uptake(model, data, ex, args.fixed_carbon_rate)
    for ex in AMINO_EXCHANGES:
        set_uptake(model, data, ex, args.amino_rate)

    set_secretion_only(model, exchange_default_bounds, "EX_etoh_e")


def apply_mixed_regime(model, data, rng, base_exchanges, args):
    apply_base_medium(model, data, base_exchanges, args.base_rate)
    apply_minn_context_caps(model, data, rng, args)

    extra_carbons = draw_subset(
        rng,
        NON_CONTEXT_FAURE_CARBON_EXCHANGES,
        max_sources=args.max_mixed_extra_carbon_sources,
        min_sources=0,
    )
    for ex in extra_carbons:
        rate = random_rate(rng, args.carbon_rate_min, args.carbon_rate_max)
        set_uptake(model, data, ex, rate)

    if rng.random() < args.mixed_supplement_probability:
        for ex in FIXED_CARBON_EXCHANGES:
            set_uptake(model, data, ex, args.fixed_carbon_rate)
        for ex in AMINO_EXCHANGES:
            set_uptake(model, data, ex, args.amino_rate)


def generate_training_sample(
    model,
    rng,
    exchange_default_bounds,
    outputs,
    input_cols,
    base_exchanges,
    regime_names,
    regime_probabilities,
    args,
):
    data = {col: 0.0 for col in input_cols}
    regime = choose_regime(rng, regime_names, regime_probabilities)

    try:
        reset_closed_medium(model, exchange_default_bounds)

        if regime == "minn":
            apply_minn_regime(model, data, rng, base_exchanges, args)
        elif regime == "faure":
            apply_faure_regime(model, data, rng, exchange_default_bounds, base_exchanges, args)
        elif regime == "mixed":
            apply_mixed_regime(model, data, rng, base_exchanges, args)
        else:
            raise ValueError(f"Unknown regime: {regime}")

        solution = solve_fluxes(
            model=model,
            flux_solver_mode=args.flux_solver_mode,
            pfba_fraction_of_optimum=args.pfba_fraction_of_optimum,
        )
        solution_status = getattr(solution, "status", "optimal")
        if solution_status != "optimal":
            return None, str(solution_status), regime

        for rxn_id in outputs:
            data[f"{rxn_id}_flux"] = float(solution.fluxes.get(rxn_id, 0.0))

        return data, "optimal", regime

    except Exception as exc:
        return None, f"error:{type(exc).__name__}", regime


def print_progress(
    sample_count,
    attempt_count,
    n_samples,
    status_counts,
    accepted_regime_counts,
    start_time,
):
    elapsed = time.time() - start_time
    feasible_rate = sample_count / max(1, attempt_count)
    nonoptimal = attempt_count - sample_count
    status_summary = ", ".join(
        f"{status}={count}"
        for status, count in status_counts.items()
        if status != "optimal"
    )
    if not status_summary:
        status_summary = "none"
    regime_summary = ", ".join(
        f"{regime}={count}" for regime, count in sorted(accepted_regime_counts.items())
    )
    print(
        f"Generated {sample_count}/{n_samples} samples "
        f"(attempts={attempt_count}, feasible_rate={feasible_rate:.3f}, "
        f"nonoptimal={nonoptimal}, {format_elapsed(elapsed)} elapsed, "
        f"time {datetime.now().strftime('%H:%M')}, statuses: {status_summary}, "
        f"accepted regimes: {regime_summary})"
    )


def main():
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not (0 < args.pfba_fraction_of_optimum <= 1):
        raise ValueError("--pfba-fraction-of-optimum must be in (0, 1]")
    if not (0 <= args.mixed_supplement_probability <= 1):
        raise ValueError("--mixed-supplement-probability must be in [0, 1]")

    integer_cap_ranges = {
        "glucose": (args.glucose_rate_min, args.glucose_rate_max),
        "oxygen": (args.oxygen_rate_min, args.oxygen_rate_max),
        "CO2 secretion": (
            args.co2_secretion_cap_min,
            args.co2_secretion_cap_max,
        ),
        "ethanol secretion": (
            args.ethanol_secretion_cap_min,
            args.ethanol_secretion_cap_max,
        ),
        "acetate secretion": (
            args.acetate_secretion_cap_min,
            args.acetate_secretion_cap_max,
        ),
    }
    for name, (min_val, max_val) in integer_cap_ranges.items():
        if min_val < 0 or max_val < min_val:
            raise ValueError(
                f"Invalid {name} integer cap range [{min_val}, {max_val}]"
            )

    rng = np.random.default_rng(args.seed)
    regime_names, regime_probabilities = normalize_regime_weights(args)
    input_cols, base_exchanges = build_input_cols(exclude_cbl1=args.exclude_cbl1)

    os.makedirs(args.data_dir, exist_ok=True)
    planned_final_filename = os.path.join(
        args.data_dir,
        f"{args.output_prefix}_{args.n_samples}_samples.csv",
    )
    if os.path.exists(planned_final_filename) and not args.overwrite_existing:
        raise FileExistsError(
            f"{planned_final_filename} already exists. Use --overwrite-existing "
            "or choose a different --output-prefix."
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_filename = os.path.join(
        args.data_dir,
        f"{args.output_prefix}_{run_stamp}_temp.csv",
    )

    model, exchange_default_bounds, outputs = load_generation_model(
        model_dir=args.model_dir,
        objective_reaction=args.objective_reaction,
        solver_timeout_seconds=args.solver_timeout_seconds,
    )
    output_cols = [f"{rxn}_flux" for rxn in outputs]
    ordered_columns = input_cols + output_cols
    validate_setup(model, input_cols, output_cols)

    max_attempts = None
    if args.max_attempt_multiplier > 0:
        max_attempts = int(np.ceil(args.n_samples * args.max_attempt_multiplier))

    print(f"Random seed: {args.seed}")
    print("Objective reaction:", model.objective)
    print(f"Flux solver mode: {args.flux_solver_mode}")
    if args.flux_solver_mode == "pfba":
        print(f"pFBA fraction_of_optimum: {args.pfba_fraction_of_optimum}")
    if args.solver_timeout_seconds > 0:
        print(f"Solver timeout: {args.solver_timeout_seconds:g} seconds")
    print("Regime probabilities:")
    for name, prob in zip(regime_names, regime_probabilities):
        print(f"  {name}: {prob:.3f}")
    print(f"Shared non-carbon base nutrient rate: {args.base_rate:g}")
    print("MINN integer context caps (inclusive min, max):")
    for name, cap_range in integer_cap_ranges.items():
        print(f"  {name}: {cap_range}")
    print("MINN context inputs:")
    for ex in MINN_CONTEXT_EXCHANGES:
        print(f"  {ex}")
    print("Faure carbon sources:")
    for ex in FAURE_CARBON_EXCHANGES:
        print(f"  {ex}")
    print(f"Input columns: {len(input_cols)}")
    print(f"Output columns: {len(output_cols)}")
    print(f"Generating {args.n_samples} shared AMN/MINN samples...")
    print(f"Output temp file: {temp_filename}")
    print(f"Planned final file: {planned_final_filename}")

    start_time = time.time()
    sample_count = 0
    attempt_count = 0
    batch = []
    status_counts = Counter()
    attempted_regime_counts = Counter()
    accepted_regime_counts = Counter()
    consecutive_failures = 0

    with open(temp_filename, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ordered_columns)

        while sample_count < args.n_samples:
            if max_attempts is not None and attempt_count >= max_attempts:
                raise RuntimeError(
                    f"Stopped after {attempt_count} attempts with "
                    f"{sample_count}/{args.n_samples} accepted samples. "
                    f"Status counts: {dict(status_counts)}"
                )

            if (
                args.solver_reset_interval > 0
                and attempt_count > 0
                and attempt_count % args.solver_reset_interval == 0
            ):
                del model
                gc.collect()
                model, exchange_default_bounds, reloaded_outputs = load_generation_model(
                    model_dir=args.model_dir,
                    objective_reaction=args.objective_reaction,
                    solver_timeout_seconds=args.solver_timeout_seconds,
                )
                if reloaded_outputs != outputs:
                    raise RuntimeError("Reaction order changed after model reload.")

            attempt_count += 1
            reported_this_attempt = False
            sample, status, regime = generate_training_sample(
                model=model,
                rng=rng,
                exchange_default_bounds=exchange_default_bounds,
                outputs=outputs,
                input_cols=input_cols,
                base_exchanges=base_exchanges,
                regime_names=regime_names,
                regime_probabilities=regime_probabilities,
                args=args,
            )
            status_counts[status] += 1
            attempted_regime_counts[regime] += 1

            if sample:
                row = [sample.get(col, 0.0) for col in ordered_columns]
                batch.append(row)
                sample_count += 1
                accepted_regime_counts[regime] += 1
                consecutive_failures = 0

                if (
                    args.progress_interval > 0
                    and sample_count % args.progress_interval == 0
                ):
                    print_progress(
                        sample_count=sample_count,
                        attempt_count=attempt_count,
                        n_samples=args.n_samples,
                        status_counts=status_counts,
                        accepted_regime_counts=accepted_regime_counts,
                        start_time=start_time,
                    )
                    reported_this_attempt = True
            else:
                consecutive_failures += 1
                if (
                    args.failure_reload_interval > 0
                    and consecutive_failures >= args.failure_reload_interval
                ):
                    print(
                        f"Reloading model after {consecutive_failures} "
                        f"consecutive nonoptimal samples."
                    )
                    del model
                    gc.collect()
                    model, exchange_default_bounds, reloaded_outputs = load_generation_model(
                        model_dir=args.model_dir,
                        objective_reaction=args.objective_reaction,
                        solver_timeout_seconds=args.solver_timeout_seconds,
                    )
                    if reloaded_outputs != outputs:
                        raise RuntimeError("Reaction order changed after model reload.")
                    consecutive_failures = 0

            if len(batch) >= args.batch_size:
                writer.writerows(batch)
                fh.flush()
                batch = []

            if (
                args.attempt_progress_interval > 0
                and attempt_count % args.attempt_progress_interval == 0
                and not reported_this_attempt
            ):
                print_progress(
                    sample_count=sample_count,
                    attempt_count=attempt_count,
                    n_samples=args.n_samples,
                    status_counts=status_counts,
                    accepted_regime_counts=accepted_regime_counts,
                    start_time=start_time,
                )

        if batch:
            writer.writerows(batch)
            fh.flush()

    final_filename = os.path.join(
        args.data_dir,
        f"{args.output_prefix}_{sample_count}_samples.csv",
    )
    if os.path.exists(final_filename) and args.overwrite_existing:
        os.replace(temp_filename, final_filename)
    else:
        os.rename(temp_filename, final_filename)

    total_time = time.time() - start_time
    print(f"\nCompleted {sample_count} samples in {format_elapsed(total_time)}")
    print(f"Attempts: {attempt_count}")
    print(f"Feasible rate: {sample_count / max(1, attempt_count):.3f}")
    print(f"Status counts: {dict(status_counts)}")
    print(f"Attempted regime counts: {dict(attempted_regime_counts)}")
    print(f"Accepted regime counts: {dict(accepted_regime_counts)}")
    print(f"Saved to {final_filename}")


if __name__ == "__main__":
    main()

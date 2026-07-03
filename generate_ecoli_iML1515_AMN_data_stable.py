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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate stable Faure-style AMN iML1515 FBA/pFBA samples. "
            "The default output naming matches the legacy AMN generator."
        )
    )
    parser.add_argument("--n-samples", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-prefix", default="iML1515_AMN_training_data")
    parser.add_argument("--overwrite-existing", action="store_true")

    parser.add_argument("--objective-reaction", default="BIOMASS_Ec_iML1515_core_75p37M")
    parser.add_argument("--flux-solver-mode", choices=["fba", "pfba"], default="fba")
    parser.add_argument("--pfba-fraction-of-optimum", type=float, default=0.999)
    parser.add_argument("--solver-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--solver-reset-interval", type=int, default=5000)
    parser.add_argument("--failure-reload-interval", type=int, default=100)
    parser.add_argument("--max-attempt-multiplier", type=float, default=20.0)

    parser.add_argument("--default-rate", type=float, default=10.0)
    parser.add_argument("--carbon-rate-min", type=float, default=0.05)
    parser.add_argument("--carbon-rate-max", type=float, default=2.2)
    parser.add_argument("--fixed-carbon-rate", type=float, default=2.2)
    parser.add_argument("--amino-rate", type=float, default=2.2)
    parser.add_argument(
        "--fixed-oxygen",
        action="store_true",
        help=(
            "Fix EX_o2_e at --default-rate. By default oxygen is sampled "
            "between --oxygen-rate-min and --oxygen-rate-max."
        ),
    )
    parser.add_argument("--oxygen-rate-min", type=float, default=1.0)
    parser.add_argument("--oxygen-rate-max", type=float, default=10.0)
    parser.add_argument("--max-carbon-sources", type=int, default=4)

    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--attempt-progress-interval", type=int, default=10000)
    return parser.parse_args()


def draw_subset(rng, exchanges, max_sources=4):
    k = rng.integers(1, min(max_sources, len(exchanges)) + 1)
    return rng.choice(exchanges, size=int(k), replace=False).tolist()


def random_rate(rng, min_val=0.1, max_val=10.0, log_uniform=False):
    if log_uniform:
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        return round(float(10 ** rng.uniform(log_min, log_max)), 2)
    return round(float(rng.uniform(min_val, max_val)), 2)


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


def generate_training_sample(
    model,
    rng,
    exchange_default_bounds,
    outputs,
    args,
):
    data = {}
    try:
        reset_closed_medium(model, exchange_default_bounds)

        carbon_subset = draw_subset(
            rng,
            CARBON_EXCHANGES,
            max_sources=args.max_carbon_sources,
        )
        for ex in carbon_subset:
            rate = random_rate(
                rng,
                min_val=args.carbon_rate_min,
                max_val=args.carbon_rate_max,
            )
            set_uptake(model, data, ex, rate)

        for ex in FIXED_CARBON_EXCHANGES:
            set_uptake(model, data, ex, args.fixed_carbon_rate)

        for ex in BASE_EXCHANGES:
            if ex == "EX_o2_e":
                if args.fixed_oxygen:
                    rate = args.default_rate
                else:
                    rate = random_rate(
                        rng,
                        min_val=args.oxygen_rate_min,
                        max_val=args.oxygen_rate_max,
                    )
                set_uptake(model, data, ex, rate)
            else:
                set_uptake(model, data, ex, args.default_rate)

        for ex in AMINO_EXCHANGES:
            set_uptake(model, data, ex, args.amino_rate)

        solution = solve_fluxes(
            model=model,
            flux_solver_mode=args.flux_solver_mode,
            pfba_fraction_of_optimum=args.pfba_fraction_of_optimum,
        )
        solution_status = getattr(solution, "status", "optimal")
        if solution_status != "optimal":
            return None, str(solution_status)

        for rxn_id in outputs:
            data[f"{rxn_id}_flux"] = float(solution.fluxes.get(rxn_id, 0.0))

        return data, "optimal"

    except Exception as exc:
        return None, f"error:{type(exc).__name__}"


def print_progress(sample_count, attempt_count, n_samples, status_counts, start_time):
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
    print(
        f"Generated {sample_count}/{n_samples} samples "
        f"(attempts={attempt_count}, feasible_rate={feasible_rate:.3f}, "
        f"nonoptimal={nonoptimal}, {format_elapsed(elapsed)} elapsed, "
        f"time {datetime.now().strftime('%H:%M')}, statuses: {status_summary})"
    )


def main():
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not (0 < args.pfba_fraction_of_optimum <= 1):
        raise ValueError("--pfba-fraction-of-optimum must be in (0, 1]")

    rng = np.random.default_rng(args.seed)
    input_cols = (
        BASE_EXCHANGES
        + FIXED_CARBON_EXCHANGES
        + AMINO_EXCHANGES
        + CARBON_EXCHANGES
    )

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

    print("Objective reaction:", model.objective)
    print(f"Flux solver mode: {args.flux_solver_mode}")
    if args.flux_solver_mode == "pfba":
        print(f"pFBA fraction_of_optimum: {args.pfba_fraction_of_optimum}")
    if args.solver_timeout_seconds > 0:
        print(f"Solver timeout: {args.solver_timeout_seconds:g} seconds")
    print(f"Obligate/default uptake rate: {args.default_rate:g}")
    print(f"Fixed glycerol uptake rate: {args.fixed_carbon_rate:g}")
    print(f"Fixed amino-acid uptake rate: {args.amino_rate:g}")
    print(
        "Oxygen uptake: "
        + (
            f"fixed at {args.default_rate:g}"
            if args.fixed_oxygen
            else f"sampled in [{args.oxygen_rate_min:g}, {args.oxygen_rate_max:g}]"
        )
    )
    print(f"Generating {args.n_samples} AMN-style samples...")
    print(f"Output temp file: {temp_filename}")
    print(f"Planned final file: {planned_final_filename}")
    print("Variable carbon sources:")
    for ex in CARBON_EXCHANGES:
        print(f"  {ex}")
    print("")

    start_time = time.time()
    sample_count = 0
    attempt_count = 0
    batch = []
    status_counts = Counter()
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
            sample, status = generate_training_sample(
                model=model,
                rng=rng,
                exchange_default_bounds=exchange_default_bounds,
                outputs=outputs,
                args=args,
            )
            status_counts[status] += 1

            if sample:
                row = [sample.get(col, 0.0) for col in ordered_columns]
                batch.append(row)
                sample_count += 1
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
    print(f"Saved to {final_filename}")


if __name__ == "__main__":
    main()

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

FIXED_BASE_EXCHANGES = [
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
    "EX_tungs_e",
    "EX_slnt_e",
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

AMINO_EXCHANGES = [
    "EX_ala__L_e",
    "EX_pro__L_e",
    "EX_thr__L_e",
    "EX_gly_e",
]

SELECTABLE_ORGANIC_EXCHANGES = [
    "EX_glc__D_e",
    "EX_xyl__D_e",
    "EX_arab__L_e",
    "EX_man_e",
    "EX_gal_e",
    "EX_fru_e",
    "EX_malt_e",
    "EX_tre_e",
    "EX_cellb_e",
    "EX_melib_e",
    "EX_lcts_e",
    "EX_rib__D_e",
    "EX_sbt__D_e",
    "EX_mnl_e",
    "EX_glyc_e",
    "EX_rmn_e",
    "EX_ac_e",
    "EX_lac__D_e",
    "EX_lac__L_e",
    "EX_pyr_e",
    "EX_succ_e",
    "EX_fum_e",
    "EX_cit_e",
    "EX_etoh_e",
    "EX_but_e",
    "EX_ppa_e",
    "EX_glcn_e",
    "EX_ala__L_e",
    "EX_pro__L_e",
    "EX_thr__L_e",
    "EX_gly_e",
]

DEFAULT_OBJECTIVE_REACTION = "BIOMASS_Ec_iML1515_core_75p37M"
DEFAULT_G_K_BETA = 0.3152078146

A_REGIME = 0
B_REGIME = 1
G_REGIME = 2
REGIME_NAMES = {
    A_REGIME: "a",
    B_REGIME: "b",
    G_REGIME: "g",
}


def create_parser(model_kind):
    if model_kind not in {"d", "e"}:
        raise ValueError("model_kind must be 'd' or 'e'")

    task_aware = model_kind == "d"
    model_label = "task-aware model D" if task_aware else "task-agnostic model E"
    parser = argparse.ArgumentParser(
        description=(
            f"Generate broad iML1515 pFBA samples for {model_label}."
        )
    )
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--output-prefix",
        default=(
            "iML1515_D_training_data"
            if task_aware
            else "iML1515_E_training_data"
        ),
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

    parser.add_argument("--g-base-rate", type=float, default=50.0)
    parser.add_argument("--g-organic-rate-min", type=float, default=0.05)
    parser.add_argument("--g-organic-rate-max", type=float, default=5.0)
    parser.add_argument("--g-oxygen-rate-min", type=float, default=1.0)
    parser.add_argument("--g-oxygen-rate-max", type=float, default=10.0)
    parser.add_argument("--g-max-organic-sources", type=int, default=8)
    parser.add_argument("--g-k-beta", type=float, default=DEFAULT_G_K_BETA)

    if task_aware:
        parser.add_argument(
            "--task-fraction",
            type=float,
            default=0.2,
            help=(
                "Total accepted-sample fraction reserved equally for exact A "
                "and B regimes. The documented 0.2 default is a pilot setting."
            ),
        )
        parser.add_argument("--a-base-rate", type=float, default=10.0)
        parser.add_argument("--a-carbon-rate-min", type=float, default=0.05)
        parser.add_argument("--a-carbon-rate-max", type=float, default=2.2)
        parser.add_argument("--a-fixed-glycerol-rate", type=float, default=2.2)
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
    return parser


def format_elapsed(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def random_uniform_rate(rng, minimum, maximum):
    return round(float(rng.uniform(minimum, maximum)), 2)


def random_log_uniform_rate(rng, minimum, maximum):
    log_rate = rng.uniform(np.log(minimum), np.log(maximum))
    return round(float(np.exp(log_rate)), 2)


def random_integer_cap(rng, minimum, maximum):
    return int(rng.integers(minimum, maximum + 1))


def draw_uniform_subset(rng, exchanges, count):
    return rng.choice(exchanges, size=count, replace=False).tolist()


def build_g_k_probabilities(max_sources, beta):
    counts = np.arange(1, max_sources + 1, dtype=np.float64)
    weights = np.exp(-beta * counts)
    return weights / weights.sum()


def draw_g_source_count(rng, max_sources, probabilities):
    return int(rng.choice(np.arange(1, max_sources + 1), p=probabilities))


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
    context_set = set(MINN_CONTEXT_EXCHANGES)
    remaining_base = [
        exchange_id
        for exchange_id in FIXED_BASE_EXCHANGES
        if exchange_id not in context_set
    ]
    remaining_organic = [
        exchange_id
        for exchange_id in SELECTABLE_ORGANIC_EXCHANGES
        if exchange_id not in context_set
    ]
    return list(MINN_CONTEXT_EXCHANGES) + remaining_base + remaining_organic


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


def validate_range(name, minimum, maximum, strictly_positive=False):
    invalid_minimum = minimum <= 0 if strictly_positive else minimum < 0
    if invalid_minimum or maximum < minimum:
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(
            f"Invalid {name} range [{minimum}, {maximum}]; values must be "
            f"{qualifier} and maximum must be at least minimum"
        )


def validate_args(args, model_kind):
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0 < args.pfba_fraction_of_optimum <= 1:
        raise ValueError("--pfba-fraction-of-optimum must be in (0, 1]")
    if args.g_base_rate < 0:
        raise ValueError("--g-base-rate must be non-negative")
    if args.g_k_beta < 0:
        raise ValueError("--g-k-beta must be non-negative")
    if not 1 <= args.g_max_organic_sources <= len(SELECTABLE_ORGANIC_EXCHANGES):
        raise ValueError(
            "--g-max-organic-sources must be between 1 and "
            f"{len(SELECTABLE_ORGANIC_EXCHANGES)}"
        )

    validate_range(
        "G organic uptake",
        args.g_organic_rate_min,
        args.g_organic_rate_max,
        strictly_positive=True,
    )
    validate_range(
        "G oxygen uptake",
        args.g_oxygen_rate_min,
        args.g_oxygen_rate_max,
    )

    if model_kind == "e":
        return

    if not 0 <= args.task_fraction <= 1:
        raise ValueError("--task-fraction must be in [0, 1]")
    if not 1 <= args.a_max_carbon_sources <= len(AMN_CARBON_EXCHANGES):
        raise ValueError(
            "--a-max-carbon-sources must be between 1 and "
            f"{len(AMN_CARBON_EXCHANGES)}"
        )
    for name, value in {
        "A base rate": args.a_base_rate,
        "A fixed glycerol rate": args.a_fixed_glycerol_rate,
        "A amino-acid rate": args.a_amino_rate,
        "B base rate": args.b_base_rate,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    validate_range(
        "A carbon uptake", args.a_carbon_rate_min, args.a_carbon_rate_max
    )
    validate_range(
        "A oxygen uptake", args.a_oxygen_rate_min, args.a_oxygen_rate_max
    )
    for exchange_id, (minimum, maximum) in build_b_cap_config(args).items():
        validate_range(f"B cap for {exchange_id}", minimum, maximum)


def build_b_cap_config(args):
    return {
        "EX_glc__D_e": (args.b_glucose_cap_min, args.b_glucose_cap_max),
        "EX_o2_e": (args.b_oxygen_cap_min, args.b_oxygen_cap_max),
        "EX_co2_e": (args.b_co2_cap_min, args.b_co2_cap_max),
        "EX_etoh_e": (args.b_ethanol_cap_min, args.b_ethanol_cap_max),
        "EX_ac_e": (args.b_acetate_cap_min, args.b_acetate_cap_max),
    }


def build_regime_schedule(n_samples, model_kind, task_fraction, rng):
    if model_kind == "e":
        return np.full(n_samples, G_REGIME, dtype=np.uint8)

    task_total = int(round(n_samples * task_fraction))
    a_count = task_total // 2
    b_count = task_total - a_count

    schedule = np.empty(n_samples, dtype=np.uint8)
    schedule[:a_count] = A_REGIME
    schedule[a_count : a_count + b_count] = B_REGIME
    schedule[a_count + b_count :] = G_REGIME
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
    reaction = model.reactions.get_by_id(exchange_id)
    reaction.lower_bound = -float(rate)
    data[exchange_id] = float(rate)


def set_secretion_cap(model, data, exchange_id, cap):
    reaction = model.reactions.get_by_id(exchange_id)
    reaction.lower_bound = 0.0
    reaction.upper_bound = float(cap)
    data[exchange_id] = float(cap)


def apply_a_regime(model, data, rng, exchange_default_bounds, args):
    reset_closed_medium(model, exchange_default_bounds)

    for exchange_id in COMMON_BASE_EXCHANGES:
        set_uptake(model, data, exchange_id, args.a_base_rate)
    set_uptake(model, data, "EX_co2_e", args.a_base_rate)

    oxygen_rate = random_uniform_rate(
        rng, args.a_oxygen_rate_min, args.a_oxygen_rate_max
    )
    set_uptake(model, data, "EX_o2_e", oxygen_rate)

    carbon_count = int(
        rng.integers(1, min(args.a_max_carbon_sources, len(AMN_CARBON_EXCHANGES)) + 1)
    )
    for exchange_id in draw_uniform_subset(
        rng, AMN_CARBON_EXCHANGES, carbon_count
    ):
        rate = random_uniform_rate(
            rng, args.a_carbon_rate_min, args.a_carbon_rate_max
        )
        set_uptake(model, data, exchange_id, rate)

    set_uptake(model, data, "EX_glyc_e", args.a_fixed_glycerol_rate)
    for exchange_id in AMINO_EXCHANGES:
        set_uptake(model, data, exchange_id, args.a_amino_rate)


def apply_b_regime(
    model,
    data,
    rng,
    exchange_default_bounds,
    b_cap_config,
    args,
):
    restore_default_medium(model, exchange_default_bounds)

    for exchange_id in COMMON_BASE_EXCHANGES + ["EX_cbl1_e"]:
        set_uptake(model, data, exchange_id, args.b_base_rate)

    for exchange_id in ("EX_glc__D_e", "EX_o2_e"):
        cap = random_integer_cap(rng, *b_cap_config[exchange_id])
        set_uptake(model, data, exchange_id, cap)

    for exchange_id in ("EX_co2_e", "EX_etoh_e", "EX_ac_e"):
        cap = random_integer_cap(rng, *b_cap_config[exchange_id])
        set_secretion_cap(model, data, exchange_id, cap)


def apply_g_regime(
    model,
    data,
    rng,
    exchange_default_bounds,
    g_k_probabilities,
    args,
):
    reset_closed_medium(model, exchange_default_bounds)

    for exchange_id in FIXED_BASE_EXCHANGES:
        if exchange_id == "EX_co2_e":
            set_secretion_cap(model, data, exchange_id, args.g_base_rate)
        else:
            set_uptake(model, data, exchange_id, args.g_base_rate)

    oxygen_rate = random_uniform_rate(
        rng, args.g_oxygen_rate_min, args.g_oxygen_rate_max
    )
    set_uptake(model, data, "EX_o2_e", oxygen_rate)

    source_count = draw_g_source_count(
        rng, args.g_max_organic_sources, g_k_probabilities
    )
    for exchange_id in draw_uniform_subset(
        rng, SELECTABLE_ORGANIC_EXCHANGES, source_count
    ):
        rate = random_log_uniform_rate(
            rng, args.g_organic_rate_min, args.g_organic_rate_max
        )
        set_uptake(model, data, exchange_id, rate)


def generate_training_sample(
    model,
    rng,
    regime,
    input_columns,
    outputs,
    exchange_default_bounds,
    g_k_probabilities,
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
        elif regime == "g":
            apply_g_regime(
                model,
                data,
                rng,
                exchange_default_bounds,
                g_k_probabilities,
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
        f"{regime}={accepted_regime_counts[regime]}" for regime in ("a", "b", "g")
    )
    print(
        f"Generated {sample_count}/{n_samples} samples "
        f"(attempts={attempt_count}, feasible_rate={feasible_rate:.3f}, "
        f"accepted regimes: {regime_summary}, elapsed={format_elapsed(elapsed)}, "
        f"time={datetime.now().strftime('%H:%M')}, "
        f"status_counts={dict(status_counts)})"
    )


def run_generation(args, model_kind):
    validate_args(args, model_kind)

    input_columns = build_input_columns()
    b_cap_config = build_b_cap_config(args) if model_kind == "d" else None
    g_k_probabilities = build_g_k_probabilities(
        args.g_max_organic_sources, args.g_k_beta
    )

    seed_sequence = np.random.SeedSequence(args.seed)
    schedule_seed, a_seed, b_seed, g_seed = seed_sequence.spawn(4)
    schedule_rng = np.random.default_rng(schedule_seed)
    regime_rngs = {
        "a": np.random.default_rng(a_seed),
        "b": np.random.default_rng(b_seed),
        "g": np.random.default_rng(g_seed),
    }
    task_fraction = args.task_fraction if model_kind == "d" else 0.0
    regime_schedule = build_regime_schedule(
        args.n_samples, model_kind, task_fraction, schedule_rng
    )
    requested_regime_counts = Counter(
        REGIME_NAMES[int(regime)] for regime in regime_schedule
    )

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

    expected_k = float(
        np.dot(
            np.arange(1, args.g_max_organic_sources + 1),
            g_k_probabilities,
        )
    )
    print(f"Model: {model_path}")
    print(f"Objective reaction: {args.objective_reaction}")
    print("Flux solver mode: pfba")
    print(f"pFBA fraction_of_optimum: {args.pfba_fraction_of_optimum}")
    print(f"Random seed: {args.seed}")
    print(f"Input columns: {len(input_columns)}")
    print(f"Output columns: {len(output_columns)}")
    print(f"Accepted-sample regime targets: {dict(requested_regime_counts)}")
    print(
        "G organic uptake: log-uniform "
        f"[{args.g_organic_rate_min:g}, {args.g_organic_rate_max:g}]"
    )
    print(
        "G oxygen uptake: continuous "
        f"[{args.g_oxygen_rate_min:g}, {args.g_oxygen_rate_max:g}]"
    )
    print(
        f"G active-source count: 1--{args.g_max_organic_sources}, "
        f"beta={args.g_k_beta:g}, expected K={expected_k:.6f}"
    )
    print(f"G fixed-base rate: {args.g_base_rate:g}")
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
                rng=regime_rngs[regime],
                regime=regime,
                input_columns=input_columns,
                outputs=outputs,
                exchange_default_bounds=exchange_default_bounds,
                g_k_probabilities=g_k_probabilities,
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
                    sample_count,
                    attempt_count,
                    args.n_samples,
                    accepted_regime_counts,
                    status_counts,
                    start_time,
                )
            elif (
                args.attempt_progress_interval > 0
                and attempt_count % args.attempt_progress_interval == 0
            ):
                print_progress(
                    sample_count,
                    attempt_count,
                    args.n_samples,
                    accepted_regime_counts,
                    status_counts,
                    start_time,
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

    if accepted_regime_counts != requested_regime_counts:
        raise RuntimeError(
            "Accepted regime counts do not match the requested schedule: "
            f"accepted={dict(accepted_regime_counts)}, "
            f"requested={dict(requested_regime_counts)}"
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


def parse_args():
    return create_parser("e").parse_args()


def main():
    run_generation(parse_args(), "e")


if __name__ == "__main__":
    main()

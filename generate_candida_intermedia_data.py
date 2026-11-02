"""Generate signed-bound/signed-flux data for the Candida intermedia pilot."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import cobra
from cobra.exceptions import OptimizationError
from cobra.flux_analysis import pfba
from cobra.io import read_sbml_model
from cobra.util.solver import linear_reaction_coefficients
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "metabolic_models" / (
    "Candida_intermedia_C04520--870.8747989420568-1002.5087414889282.sbml"
)
BASAL_EXCHANGES = (
    "UF02549_E", "UF02765_E", "UF02997_E", "UF03268_E",
    "UF03314_E", "UF03376_E", "UF03456_E", "UF03474_E",
)
OXYGEN_EXCHANGE = "UF03382_E"
CARBON_EXCHANGES = {
    "arabinose": "UF03192_E",
    "glucose": "UF03288_E",
    "galactose": "UF03281_E",
    "xylose": "UF03503_E",
    "manose": "UF03359_E",
    "fructose": "UF03275_E",
    "lactate": "UF02967_E",  # Provisional L-lactate only, not experimental isomer evidence.
    "acetate": "UF03160_E",
}
GLUCOSE_EXCHANGE = CARBON_EXCHANGES["glucose"]
OPTIONAL_CARBON_EXCHANGES = tuple(
    reaction_id for reaction_id in CARBON_EXCHANGES.values()
    if reaction_id != GLUCOSE_EXCHANGE
)
INPUT_EXCHANGES = BASAL_EXCHANGES + (OXYGEN_EXCHANGE,) + tuple(CARBON_EXCHANGES.values())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("training", "test"), default="training")
    parser.add_argument("--n-samples", type=int, help="Accepted rows; default training=100000, test=50000.")
    parser.add_argument("--seed", type=int, help="Default training=42, test=9.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-prefix", help="Default candida_intermedia_{dataset}_data.")
    parser.add_argument("--max-additional-carbons", type=int, default=7)
    parser.add_argument("--carbon-min", type=float, default=0.1, help="Positive uptake magnitude.")
    parser.add_argument("--carbon-max", type=float, default=10.0)
    parser.add_argument("--oxygen-min", type=float, default=0.1)
    parser.add_argument("--oxygen-max", type=float, default=10.0)
    parser.add_argument("--uptake-distribution", choices=("uniform", "loguniform"), default="uniform")
    parser.add_argument("--flux-solver-mode", choices=("fba", "pfba"), default="pfba")
    parser.add_argument("--pfba-fraction-of-optimum", type=float, default=0.999)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--max-attempt-multiplier", type=int, default=20)
    args = parser.parse_args(argv)
    if args.n_samples is None:
        args.n_samples = 100_000 if args.dataset == "training" else 50_000
    if args.seed is None:
        args.seed = 42 if args.dataset == "training" else 9
    if args.output_prefix is None:
        args.output_prefix = f"candida_intermedia_{args.dataset}_data"
    if args.n_samples <= 0 or args.seed < 0:
        parser.error("--n-samples must be positive and --seed nonnegative")
    if not 0 <= args.max_additional_carbons <= len(OPTIONAL_CARBON_EXCHANGES):
        parser.error("--max-additional-carbons must be in [0, 7]")
    for name in ("carbon", "oxygen"):
        low, high = getattr(args, f"{name}_min"), getattr(args, f"{name}_max")
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
            parser.error(f"--{name}-min/max must be finite with 0 < min <= max")
    if not 0 < args.pfba_fraction_of_optimum <= 1:
        parser.error("--pfba-fraction-of-optimum must be in (0, 1]")
    if args.progress_interval < 0 or args.max_attempt_multiplier < 1:
        parser.error("--progress-interval must be nonnegative; --max-attempt-multiplier positive")
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        parser.error("--output-prefix must be a filename prefix, not a path")
    return args


def validate_model(model):
    exchanges = {reaction.id: reaction for reaction in model.exchanges}
    for reaction_id in INPUT_EXCHANGES:
        if reaction_id not in exchanges:
            raise ValueError(f"Required exchange absent: {reaction_id}")
        reaction = exchanges[reaction_id]
        if len(reaction.metabolites) != 1 or next(iter(reaction.metabolites.values())) != -1:
            raise ValueError(f"Expected negative uptake convention for {reaction_id}")
        if reaction.upper_bound < 0:
            raise ValueError(f"Cannot close uptake while preserving upper bound for {reaction_id}")
    if model.objective.direction != "max" or not linear_reaction_coefficients(model):
        raise ValueError("Expected a nonempty maximization objective in the supplied model")


def draw_uptake(rng, low, high, distribution):
    if distribution == "uniform":
        return float(rng.uniform(low, high))
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def draw_medium(rng, args):
    """Return all 17 signed lower bounds; selected magnitudes are never rounded."""
    medium = dict.fromkeys(INPUT_EXCHANGES, 0.0)
    medium.update({reaction_id: -1000.0 for reaction_id in BASAL_EXCHANGES})
    count = int(rng.integers(args.max_additional_carbons + 1))
    optional = rng.choice(OPTIONAL_CARBON_EXCHANGES, size=count, replace=False)
    for reaction_id in (GLUCOSE_EXCHANGE, *optional):
        medium[reaction_id] = -draw_uptake(
            rng, args.carbon_min, args.carbon_max, args.uptake_distribution
        )
    medium[OXYGEN_EXCHANGE] = -draw_uptake(
        rng, args.oxygen_min, args.oxygen_max, args.uptake_distribution
    )
    return medium


def solve_medium(model, medium, outputs, args):
    """Solve one condition; caller has closed all exchange uptake beforehand."""
    with model:
        for reaction_id, lower_bound in medium.items():
            model.reactions.get_by_id(reaction_id).lower_bound = lower_bound
        try:
            if args.flux_solver_mode == "pfba":
                solution = pfba(model, fraction_of_optimum=args.pfba_fraction_of_optimum)
            else:
                solution = model.optimize()
        except OptimizationError:
            return None, "optimization_error"
        if solution.status != "optimal":
            return None, solution.status
        fluxes = solution.fluxes.loc[outputs].to_numpy(dtype=float)
        if not np.isfinite(fluxes).all():
            return None, "nonfinite_flux"
        return fluxes, "optimal"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_dataset(args):
    model = read_sbml_model(str(args.model_path))
    validate_model(model)
    # Stable ordered full-reaction vocabulary, one signed net-flux token per ID.
    outputs = sorted(reaction.id for reaction in model.reactions)
    columns = list(INPUT_EXCHANGES) + [f"{reaction_id}_flux" for reaction_id in outputs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y-%m-%d}_{args.output_prefix}_{args.n_samples}_samples"
    final_path = args.output_dir / f"{stem}.csv"
    temp_path = args.output_dir / f"{stem}_temp.csv"
    metadata_path = args.output_dir / f"{stem}.json"
    for path in (final_path, temp_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose another --output-prefix")
    objective = {r.id: float(c) for r, c in linear_reaction_coefficients(model).items()}
    metadata = {
        "schema_version": 1,
        "status": "running",
        "organism": "Candida intermedia C04520",
        "model_file": args.model_path.name,
        "model_sha256": file_sha256(args.model_path),
        "model_id": model.id,
        "generator_file": Path(__file__).name,
        "generator_sha256": file_sha256(__file__),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cobra_version": cobra.__version__,
        "numpy_version": np.__version__,
        "rng": "numpy.random.default_rng (PCG64)",
        "solver_interface": model.solver.interface.__name__,
        "solver_tolerances": {
            name: getattr(model.solver.configuration.tolerances, name)
            for name in ("feasibility", "optimality", "integrality")
            if hasattr(model.solver.configuration.tolerances, name)
        },
        "objective": objective,
        "objective_direction": model.objective.direction,
        "input_columns": list(INPUT_EXCHANGES),
        "output_reaction_ids": outputs,
        "output_columns": columns[len(INPUT_EXCHANGES):],
        "input_token_indices": [outputs.index(rid) for rid in INPUT_EXCHANGES],
        "input_semantics": "signed exchange lower bounds; zero for unselected carbon",
        "target_semantics": "signed net reaction fluxes; no reverse-token splitting",
        "flux_units": "native supplied SBML flux units; not experimental concentrations",
        "basal_lower_bounds": dict.fromkeys(BASAL_EXCHANGES, -1000.0),
        "carbon_exchange_map": CARBON_EXCHANGES,
        "oxygen_exchange": OXYGEN_EXCHANGE,
        "mandatory_glucose": True,
        "additional_carbon_count": "discrete uniform 0..K; distinct subset uniform conditional on count",
        "lactate_policy": "L-only provisional; D-lactate uptake closed; experimental isomer unknown",
        "other_exchange_uptake": "closed; original secretion upper bounds retained",
        "reaction_upper_bounds": {r.id: r.upper_bound for r in model.reactions},
        "acceptance": "all optimal finite solutions, including zero growth; retry failures",
        "split_policy": "training file is for later train/validation split; test generated separately",
        "started_at": datetime.now().astimezone().isoformat(),
    }
    with metadata_path.open("x") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")
    rng = np.random.default_rng(args.seed)
    accepted = attempts = 0
    rejected = Counter()
    start = time.monotonic()
    print(f"Generating {args.n_samples} {args.dataset} rows; seed={args.seed}; "
          f"{args.flux_solver_mode.upper()}; {len(INPUT_EXCHANGES)} inputs, {len(outputs)} targets", flush=True)
    try:
        with temp_path.open("x", newline="") as handle, model:
            # model.medium closes uptake in either exchange orientation and restores on exit.
            model.medium = {}
            writer = csv.writer(handle)
            writer.writerow(columns)
            while accepted < args.n_samples:
                if attempts >= args.n_samples * args.max_attempt_multiplier:
                    raise RuntimeError(f"Attempt limit reached: {accepted}/{args.n_samples} rows, {dict(rejected)}")
                attempts += 1
                medium = draw_medium(rng, args)
                fluxes, status = solve_medium(model, medium, outputs, args)
                if fluxes is None:
                    rejected[status] += 1
                    continue
                writer.writerow([medium[rid] for rid in INPUT_EXCHANGES] + fluxes.tolist())
                accepted += 1
                if args.progress_interval and accepted % args.progress_interval == 0:
                    hours, remainder = divmod(int(time.monotonic() - start), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f"Generated {accepted}/{args.n_samples} | attempts={attempts} | "
                          f"elapsed={hours:02d}:{minutes:02d}:{seconds:02d} | time={datetime.now():%H:%M}", flush=True)
        temp_path.rename(final_path)
        metadata["status"] = "complete"
    except BaseException:
        metadata["status"] = "incomplete"
        raise
    finally:
        metadata.update({
            "accepted_samples": accepted, "attempts": attempts,
            "rejected_by_status": dict(rejected),
            "elapsed_seconds": time.monotonic() - start,
            "data_file": final_path.name if metadata["status"] == "complete" else temp_path.name,
            "finished_at": datetime.now().astimezone().isoformat(),
        })
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved {final_path}\nProvenance: {metadata_path}", flush=True)
    return final_path


def main(argv=None):
    return generate_dataset(parse_args(argv))


if __name__ == "__main__":
    main()

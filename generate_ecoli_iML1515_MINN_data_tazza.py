import csv
import os
import time
from datetime import datetime

import warnings
import numpy as np
from cobra.io import read_sbml_model
from cobra.flux_analysis import pfba

warnings.filterwarnings("ignore", message="Solver status is 'infeasible'")


SECRETION_CONTEXT_EXCHANGES = ["EX_co2_e", "EX_etoh_e", "EX_ac_e"]


def random_integer_cap(min_val, max_val):
    """Draw an integer cap uniformly from the inclusive range [min_val, max_val]."""
    if not isinstance(min_val, int) or not isinstance(max_val, int):
        raise TypeError("Integer cap endpoints must be integers.")
    if min_val < 0 or max_val < min_val:
        raise ValueError(f"Invalid integer cap range [{min_val}, {max_val}]")
    return int(np.random.randint(min_val, max_val + 1))


def generate_training_sample(
    context_exchanges,
    base_exchanges,
    outputs,
    default_rate,
    sampled_cap_config,
    exchange_default_bounds,
    flux_solver_mode,
    pfba_fraction_of_optimum,
):
    data = {}
    try:
        # Reset exchange bounds to model defaults at every sample.
        for rxn in model.exchanges:
            lb, ub = exchange_default_bounds[rxn.id]
            rxn.lower_bound = lb
            rxn.upper_bound = ub

        # Tazza-style reservoir inputs: glucose is a sampled uptake cap;
        # ethanol and acetate are independently sampled secretion caps.
        for ex in context_exchanges:
            rxn = model.reactions.get_by_id(ex)
            min_v, max_v = sampled_cap_config[ex]
            rate = random_integer_cap(min_v, max_v)

            if ex == "EX_glc__D_e":
                # COBRA uptake is negative; the CSV stores a positive cap.
                rxn.lower_bound = -rate
            elif ex in SECRETION_CONTEXT_EXCHANGES:
                # COBRA secretion is positive. Disallow uptake and cap secretion.
                rxn.lower_bound = 0.0
                rxn.upper_bound = rate
            else:
                raise ValueError(f"Unhandled context exchange: {ex}")

            data[ex] = rate

        # Base exchanges. Oxygen is a sampled uptake cap and CO2 is an
        # independently sampled secretion cap; the remaining medium inputs
        # have fixed availability.
        for ex in base_exchanges:
            if ex == "EX_o2_e":
                min_v, max_v = sampled_cap_config[ex]
                rate = random_integer_cap(min_v, max_v)
                rxn = model.reactions.get_by_id(ex)
                rxn.lower_bound = -rate
                data[ex] = rate
            elif ex in SECRETION_CONTEXT_EXCHANGES:
                min_v, max_v = sampled_cap_config[ex]
                rate = random_integer_cap(min_v, max_v)
                rxn = model.reactions.get_by_id(ex)
                rxn.lower_bound = 0.0
                rxn.upper_bound = rate
                data[ex] = rate
            else:
                # Other base exchanges are fixed medium-availability inputs;
                # their realized fluxes are still written to *_flux outputs.
                model.reactions.get_by_id(ex).lower_bound = -default_rate
                data[ex] = default_rate

        if flux_solver_mode == "pfba":
            solution = pfba(model, fraction_of_optimum=pfba_fraction_of_optimum)
        elif flux_solver_mode == "fba":
            solution = model.optimize()
        else:
            raise ValueError("flux_solver_mode must be 'pfba' or 'fba'")

        solution_status = getattr(solution, "status", "optimal")
        if solution_status != "optimal":
            return None

        # The unsuffixed context columns retain the sampled caps. Realized
        # exchange fluxes, which can be below those caps, are written to the
        # corresponding *_flux output columns with all other reaction fluxes.
        for rxn_id in outputs:
            data[f"{rxn_id}_flux"] = solution.fluxes.get(rxn_id, 0.0)

        return data

    except Exception as exc:
        print(f"Error in generate_training_sample: {exc}")
        return None


if __name__ == "__main__":
    np.random.seed(42)

    n_samples = 500000
    default_rate = 50
    batch_size = 500
    objective_variant = "core"  # "core" or "wt"
    flux_solver_mode = "pfba"  # "pfba" or "fba"
    pfba_fraction_of_optimum = 0.999

    # Load the E. coli iML1515 metabolic model.
    model_dir = "./models"
    model = read_sbml_model(os.path.join(model_dir, "iML1515.xml"))

    objective_map = {
        "core": "BIOMASS_Ec_iML1515_core_75p37M",
        "wt": "BIOMASS_Ec_iML1515_WT_75p37M",
    }
    if objective_variant not in objective_map:
        raise ValueError("objective_variant must be 'core' or 'wt'")
    if flux_solver_mode not in ["pfba", "fba"]:
        raise ValueError("flux_solver_mode must be 'pfba' or 'fba'")
    if not (0 < pfba_fraction_of_optimum <= 1):
        raise ValueError("pfba_fraction_of_optimum must be in (0, 1]")
    model.objective = objective_map[objective_variant]
    exchange_default_bounds = {
        rxn.id: (rxn.lower_bound, rxn.upper_bound) for rxn in model.exchanges
    }

    print(f"Objective variant: {objective_variant}")
    print(f"Flux solver mode: {flux_solver_mode}")
    if flux_solver_mode == "pfba":
        print(f"pFBA fraction_of_optimum: {pfba_fraction_of_optimum}")
    print("Objective reaction:", model.objective)

    # Tazza-style MINN reservoir inputs. All five exchange caps are sampled.
    context_exchanges = ["EX_glc__D_e", "EX_etoh_e", "EX_ac_e"]

    # Rounded integer envelopes around the ranges observed in the 29-sample
    # Ishii dataset. The deliberately wider ranges prevent training only on
    # the exact experimental extrema. Uptake and secretion caps are both
    # represented as positive CSV inputs.
    sampled_cap_config = {
        "EX_glc__D_e": (1, 15),
        "EX_o2_e": (1, 20),
        "EX_co2_e": (0, 15),
        "EX_etoh_e": (0, 1),
        "EX_ac_e": (0, 3),
    }

    base_exchanges = [
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

        "EX_cbl1_e",  # harmless for core; required if objective_variant="wt"
    ]

    print(f"Generating {n_samples} {flux_solver_mode.upper()} training samples...\n")
    print("Sampled integer reservoir cap configuration (inclusive min, max):")
    for ex_id in [
        "EX_glc__D_e",
        "EX_o2_e",
        "EX_co2_e",
        "EX_etoh_e",
        "EX_ac_e",
    ]:
        print(f"  {ex_id}: {sampled_cap_config[ex_id]}")
    print("Glucose and oxygen are uptake caps; CO2, ethanol, and acetate are secretion caps.")

    outputs = [rxn.id for rxn in model.reactions]
    input_cols = context_exchanges + base_exchanges
    output_cols = [f"{rxn}_flux" for rxn in outputs]
    ordered_columns = input_cols + output_cols

    os.makedirs("./data", exist_ok=True)
    today = datetime.today().strftime("%Y-%m-%d")
    temp_filename = f"./data/{today}_iML1515_MINN_tazza_training_data_temp.csv"
    start_time = time.time()

    sample_count = 0
    attempt_count = 0
    batch = []

    with open(temp_filename, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ordered_columns)

        while sample_count < n_samples:
            attempt_count += 1
            sample = generate_training_sample(
                context_exchanges=context_exchanges,
                base_exchanges=base_exchanges,
                outputs=outputs,
                default_rate=default_rate,
                sampled_cap_config=sampled_cap_config,
                exchange_default_bounds=exchange_default_bounds,
                flux_solver_mode=flux_solver_mode,
                pfba_fraction_of_optimum=pfba_fraction_of_optimum,
            )
            if sample:
                row = [sample.get(col, 0.0) for col in ordered_columns]
                batch.append(row)
                sample_count += 1
                if sample_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    hours, rem = divmod(int(elapsed), 3600)
                    minutes, _ = divmod(rem, 60)
                    now_str = datetime.now().strftime("%H:%M")
                    feasible_rate = sample_count / max(1, attempt_count)
                    print(
                        f"Generated {sample_count}/{n_samples} samples "
                        f"(attempts={attempt_count}, feasible_rate={feasible_rate:.3f}, "
                        f"{hours}h {minutes}m elapsed, time {now_str})"
                    )

            if len(batch) >= batch_size:
                writer.writerows(batch)
                fh.flush()
                batch = []

        if batch:
            writer.writerows(batch)

    final_filename = (
        f"./data/{today}_iML1515_MINN_tazza_training_data_"
        f"{sample_count}_samples.csv"
    )
    os.rename(temp_filename, final_filename)

    total_time = time.time() - start_time
    total_hours, rem = divmod(int(total_time), 3600)
    total_minutes, _ = divmod(rem, 60)
    print(f"\nCompleted {sample_count} samples in {total_hours}h {total_minutes}m")
    print(f"Saved to {final_filename}")

import csv
import os
import time
from datetime import datetime

import warnings
import numpy as np
from cobra.io import read_sbml_model
from cobra.flux_analysis import pfba

warnings.filterwarnings("ignore", message="Solver status is 'infeasible'")


def random_rate(min_val=0.1, max_val=10.0, log_uniform=False):
    """Draw a random rate between min and max (uniform or log-uniform)."""
    if log_uniform:
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        return round(float(10 ** np.random.uniform(log_min, log_max)), 2)
    return round(float(np.random.uniform(min_val, max_val)), 2)


def generate_training_sample(
    variable_carbon_exchanges,
    base_exchanges,
    outputs,
    default_rate,
    sampled_rate_config,
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

        # Variable exchanges.
        for ex in variable_carbon_exchanges:
            min_v, max_v, log_u = sampled_rate_config[ex]
            rate = random_rate(min_val=min_v, max_val=max_v, log_uniform=log_u)
            rxn = model.reactions.get_by_id(ex)
            if ex == "EX_glc__D_e":
                # Uptake: allow import via negative lower bound.
                rxn.lower_bound = -rate
            else:
                # EX_etoh_e / EX_ac_e: secretion cap via upper bound.
                rxn.lower_bound = 0.0
                rxn.upper_bound = rate
            data[ex] = rate

        # Base exchanges
        # EX_o2_e is sampled as uptake, EX_co2_e as secretion cap.
        for ex in base_exchanges:
            if ex == "EX_o2_e":
                min_v, max_v, log_u = sampled_rate_config[ex]
                rate = random_rate(min_val=min_v, max_val=max_v, log_uniform=log_u)
                rxn = model.reactions.get_by_id(ex)
                rxn.lower_bound = -rate
                data[ex] = rate
            elif ex == "EX_co2_e":
                min_v, max_v, log_u = sampled_rate_config[ex]
                rate = random_rate(min_val=min_v, max_val=max_v, log_uniform=log_u)
                rxn = model.reactions.get_by_id(ex)
                rxn.lower_bound = 0.0
                rxn.upper_bound = rate
                data[ex] = rate
            else:
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

        for rxn_id in outputs:
            data[f"{rxn_id}_flux"] = solution.fluxes.get(rxn_id, 0.0)

        return data

    except Exception as exc:
        print(f"Error in generate_training_sample: {exc}")
        return None


if __name__ == "__main__":
    np.random.seed(42) # 9 test

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

    # MINN-reservoir variable exchanges.
    variable_carbon_exchanges = ["EX_glc__D_e", "EX_etoh_e", "EX_ac_e"]
    sampled_rate_config = {
        "EX_glc__D_e": (1.0, 15.0, False),  # uniform
        "EX_o2_e": (1.0, 30.0, False),      # uniform
        "EX_co2_e": (0.1, 30.0, False),     # uniform (secretion cap)
        "EX_etoh_e": (1e-3, 1.0, True),     # log-uniform (secretion cap)
        "EX_ac_e": (1e-3, 4.0, True),       # log-uniform (secretion cap)
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

        "EX_cbl1_e", # harmless for core; required if objective_variant="wt"
    ]

    print(f"Generating {n_samples} {flux_solver_mode.upper()} training samples...\n")
    print("Sampling configuration (min, max, log_uniform):")
    for ex_id in ["EX_glc__D_e", "EX_o2_e", "EX_co2_e", "EX_etoh_e", "EX_ac_e"]:
        print(f"  {ex_id}: {sampled_rate_config[ex_id]}")
    outputs = [rxn.id for rxn in model.reactions]
    input_cols = variable_carbon_exchanges + base_exchanges
    output_cols = [f"{rxn}_flux" for rxn in outputs]
    ordered_columns = input_cols + output_cols

    os.makedirs("./data", exist_ok=True)
    today = datetime.today().strftime("%Y-%m-%d")
    temp_filename = f"./data/{today}_iML1515_MINN_training_data_temp.csv"
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
                variable_carbon_exchanges=variable_carbon_exchanges,
                base_exchanges=base_exchanges,
                outputs=outputs,
                default_rate=default_rate,
                sampled_rate_config=sampled_rate_config,
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

    final_filename = f"./data/{today}_iML1515_MINN_training_data_{sample_count}_samples.csv"
    os.rename(temp_filename, final_filename)

    total_time = time.time() - start_time
    total_hours, rem = divmod(int(total_time), 3600)
    total_minutes, _ = divmod(rem, 60)
    print(f"\nCompleted {sample_count} samples in {total_hours}h {total_minutes}m")
    print(f"Saved to {final_filename}")

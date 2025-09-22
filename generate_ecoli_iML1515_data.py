import os
import csv
import numpy as np
import pandas as pd
from cobra.io import read_sbml_model
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message="Solver status is 'infeasible'")


def draw_subset(exchanges, max_sources=5):
    """
    Randomly draw between 1 and max_sources (default=5) exchanges.
    """
    k = np.random.randint(1, min(max_sources, len(exchanges)) + 1)
    return np.random.choice(exchanges, size=k, replace=False).tolist()

def random_rate(min_val=0.1, max_val=10.0, log_uniform=False):
    """Draw uptake rate between min and max.
    
    Args:
        min_val: Minimum uptake rate
        max_val: Maximum uptake rate
        log_uniform: If True, sample log-uniformly; else sample uniformly
    """
    if log_uniform:
        # Log-uniform sampling in [min_val, max_val]
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        return round(float(10 ** np.random.uniform(log_min, log_max)), 2)
    else:
        # Uniform sampling
        return round(np.random.uniform(min_val, max_val), 2)

def generate_training_sample(carbon_subset, base_exchanges, outputs, default_rate):
    data = {}
    try:
        # Reset all exchanges
        for rxn in model.exchanges:
            rxn.lower_bound = 0.0

        # Set uptake rates for selected carbon sources
        for ex in carbon_subset:
            rate = random_rate(min_val=0.1, max_val=carbon_exhange_rate, log_uniform=False)
            model.reactions.get_by_id(ex).lower_bound = -rate
            data[ex] = rate

        # Set base exchange rates
        for ex in base_exchanges:
            if ex == "EX_o2_e":
                o2_rate = random_rate(min_val=1, max_val=default_rate, log_uniform=False)
                model.reactions.get_by_id("EX_o2_e").lower_bound = -o2_rate
                data["EX_o2_e"] = o2_rate
            else:
                model.reactions.get_by_id(ex).lower_bound = -default_rate
                data[ex] = default_rate

        # Run FBA
        solution = model.optimize()
        if solution.status != 'optimal':
            return None
        
        #print(f"Sample solution: objective={solution.objective_value:.6f}")

        for rxn_id in outputs:
            data[rxn_id + "_flux"] = solution.fluxes.get(rxn_id, 0.0)

        return data
    
    except Exception as e:
        print(f"Error in generate_training_sample: {e}")
        return None

if __name__ == "__main__":
    np.random.seed(42)

    n_samples = 100000
    default_rate = 50
    carbon_exhange_rate = 10 # 2.2 to match experimental set (Faure et al 2023): bad choice!
    batch_size = 500

    # Load the E. coli iML1515 metabolic model
    model_dir ="./models"
    model = read_sbml_model(os.path.join(model_dir, "iML1515.xml"))
    #model.objective = "BIOMASS_Ec_iML1515_WT_75p37M"

    print("Objective reaction:", model.objective)

    carbon_exchanges = [
        'EX_glc__D_e',   # D-Glucose
        'EX_rib__D_e',   # Ribose
        'EX_malt_e',     # Maltose
        'EX_melib_e',    # Melibiose
        'EX_tre_e',      # Trehalose
        'EX_fru_e',      # Fructose
        'EX_gal_e',      # Galactose
        'EX_ac_e',       # Acetate
        'EX_lac__D_e',   # D-Lactate
        'EX_succ_e',     # Succinate
        'EX_pyr_e',      # Pyruvate
    ]

    base_exchanges = [
        'EX_pi_e', 
        'EX_co2_e', 
        'EX_h_e', 
        'EX_mn2_e', 
        'EX_fe2_e', 
        'EX_zn2_e', 
        'EX_mg2_e', 
        'EX_ca2_e', 
        'EX_ni2_e', 
        'EX_cu2_e', 
        'EX_cobalt2_e', 
        'EX_h2o_e', 
        'EX_mobd_e', 
        'EX_so4_e', 
        'EX_nh4_e', 
        'EX_k_e', 
        'EX_na1_e', 
        'EX_cl_e', 
        'EX_o2_e',

        # To match experimental set (Faure et al 2023):
        # NOTE: These can't be included in base exchanges because they can act as sole carbon sources.
        # As a result the effect of variable carbon sources are negligible!
        # 'EX_ala__L_e', # Alanine
        # 'EX_pro__L_e', # Proline 
        # 'EX_thr__L_e', # Threonine
        # 'EX_gly_e', # Glycine
    ]

    print(f"Generating {n_samples} FBA training samples...\n")
    outputs = [rxn.id for rxn in model.reactions]
    input_cols = carbon_exchanges + base_exchanges
    output_cols = [f"{rxn}_flux" for rxn in outputs]
    ordered_columns = input_cols + output_cols

    # Prepare output file
    os.makedirs("./data", exist_ok=True)
    today = datetime.today().strftime('%Y-%m-%d')
    temp_filename = f"./data/{today}_iML1515_training_data_temp.csv"
    start_time = time.time()

    sample_count = 0
    batch = []

    with open(temp_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ordered_columns)
        
        for i in range(n_samples):
            carbon_subset = draw_subset(carbon_exchanges)
            
            sample = generate_training_sample(carbon_subset, base_exchanges, outputs, default_rate)
            if sample:
                row = [sample.get(col, 0.0) for col in ordered_columns]
                batch.append(row)
                sample_count += 1

            # Write batch
            if len(batch) >= batch_size:
                writer.writerows(batch)
                f.flush()
                batch = []
                elapsed = time.time() - start_time
                minutes, seconds = divmod(int(elapsed), 60)
                print(f"Generated {sample_count}/{n_samples} samples "
                    f"({minutes}m {seconds}s elapsed)")

        # Write remaining batch
        if batch:
            writer.writerows(batch)

    # Rename with actual sample count
    final_filename = f"./data/{today}_iML1515_training_data_{sample_count}_samples.csv"
    os.rename(temp_filename, final_filename)
    
    total_time = time.time() - start_time
    print(f"\nCompleted {sample_count} samples in {total_time:.2f} seconds")
    print(f"Saved to {final_filename}")
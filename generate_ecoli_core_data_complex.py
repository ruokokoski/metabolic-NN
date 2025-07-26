import os
import numpy as np
import pandas as pd
from cobra.io import load_model
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message="Solver status is 'infeasible'")

def draw_subset(exchanges, selection_rate=0.5):
    n = len(exchanges)
    k = max(1, np.random.binomial(n, selection_rate))
    return np.random.choice(exchanges, size=k, replace=False)

def draw_nitrogen(nitrogen_exchanges):
    """Draw one nitrogen source"""
    return np.random.choice(nitrogen_exchanges)

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

def generate_training_sample(carbon_subset, nitrogen_subset, outputs):
    data = {}
    with model:
        # Reset all exchanges
        for rxn in model.exchanges:
            rxn.lower_bound = 0.0

        # Set uptake rates for selected carbon sources
        for ex in carbon_subset:
            rate = random_rate(log_uniform=True)
            model.reactions.get_by_id(ex).lower_bound = -rate
            data[ex] = rate

        # Set uptake rates for selected nitrogen sources
        for ex in nitrogen_subset:
            rate = random_rate(1, default_rate)
            model.reactions.get_by_id(ex).lower_bound = -rate
            data[ex] = rate

        # Set variable oxygen uptake
        #o2_rate = round(np.random.uniform(0.1, default_rate), 2)
        o2_rate = random_rate(0.1, default_rate)
        model.reactions.get_by_id("EX_o2_e").lower_bound = -o2_rate
        data["EX_o2_e"] = o2_rate

        # Set base exchange rates
        # Leave CO2, H+, and H2O unconstrained because they are usually byproducts or balancing species
        # Constraining or varying them can cause infeasible or biologically unrealistic fluxes
        for ex in ['EX_co2_e', 'EX_h_e', 'EX_h2o_e']:
            model.reactions.get_by_id(ex).lower_bound = -default_rate
            data[ex] = default_rate

        # Make phosphate (EX_pi_e) variable, as its availability can limit growth and vary naturally
        pi_rate = random_rate(0.1, default_rate)
        model.reactions.get_by_id('EX_pi_e').lower_bound = -pi_rate
        data['EX_pi_e'] = pi_rate

        # Run FBA
        solution = model.optimize()
        if solution.status != 'optimal':
            return None

        for rxn_id in outputs:
            data[rxn_id + "_flux"] = solution.fluxes.get(rxn_id, 0.0)

        return data

if __name__ == "__main__":
    np.random.seed(42)
    default_rate = 50
    n_samples = 100000

    # Load the simplified E. coli metabolic model
    model = load_model("textbook")

    carbon_exchanges = [
        'EX_glc__D_e',   # D-Glucose
        'EX_fru_e',      # Fructose
        'EX_lac__D_e',   # D-Lactate
        'EX_pyr_e',      # Pyruvate
        'EX_ac_e',       # Acetate
        'EX_akg_e',      # 2-Oxoglutarate
        'EX_succ_e',     # Succinate
        'EX_fum_e',      # Fumarate
        'EX_mal__L_e',   # L-Malate
        'EX_etoh_e',     # Ethanol
        'EX_acald_e',    # Acetaldehyde
        'EX_for_e',      # Formate (byproduct of anaerobic fermentation)
    ]

    # In natural environments, microbes usually access a dominant nitrogen source at a time
    nitrogen_exchanges = [
        'EX_gln__L_e',   # L-Glutamine
        'EX_glu__L_e',   # L-Glutamate
        'EX_nh4_e'       # Ammonia
    ]

    base_exchanges = [
        'EX_co2_e',
        'EX_h_e',
        'EX_h2o_e',
        'EX_nh4_e',      # Ammonia
        'EX_o2_e',
        'EX_pi_e',       # Phosphate (essential)
    ]

    outputs = [rxn.id for rxn in model.reactions]
    #print([rxn.id for rxn in model.reactions])
    #print(len(model.reactions))

    training_data = []
    start_time = time.time()

    print(f"Generating {n_samples} FBA training samples...\n")
    sample_count = 0

    for _ in range(n_samples):
        carbon_subset = draw_subset(carbon_exchanges)
        nitrogen_subset = draw_subset(nitrogen_exchanges)

        sample = generate_training_sample(carbon_subset, nitrogen_subset, outputs)
        if sample:
            training_data.append(sample)
            sample_count += 1
            if sample_count % 1000 == 0:
                print(f"Progress: {sample_count}/{n_samples}")

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    input_cols = (
        carbon_exchanges + 
        ['EX_gln__L_e', 'EX_glu__L_e'] +
        base_exchanges
    )
    output_cols = [f"{rxn}_flux" for rxn in outputs]
    ordered_columns = input_cols + output_cols

    df = pd.DataFrame(training_data).reindex(columns=ordered_columns, fill_value=0.0)
                                             
    # Save data
    os.makedirs("./data", exist_ok=True)
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f"./data/{today}_full_training_data_{len(df)}_samples.csv"
    df.to_csv(filename, index=False)
    
    print(f"Saved {len(df)} samples to {filename}.")

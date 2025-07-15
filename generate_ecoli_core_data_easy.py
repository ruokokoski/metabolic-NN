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

def random_rate(min_val=0.1, max_val=10.0):
    """Draw uptake rate between min and max."""
    # return round(np.random.uniform(min_val, max_val), 2) # uniform
    return round(float(10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))), 2)

def generate_training_sample(carbon_subset, outputs):
    data = {}
    with model:
        # Reset all exchanges
        #for rxn in model.exchanges:
        #    rxn.lower_bound = 0.0
        for rxn in carbon_exchanges:
            if rxn in model.reactions:
                model.reactions.get_by_id(rxn).lower_bound = 0.0

        # Set uptake rates for selected carbon sources
        for ex in carbon_subset:
            rate = random_rate()
            model.reactions.get_by_id(ex).lower_bound = -rate
            data[ex] = rate

        # Add base exchange uptakes with default_rate to data dict
        for ex in base_exchanges:
            data[ex] = default_rate

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
        'EX_gln__L_e',   # L-Glutamine
        'EX_glu__L_e',   # L-Glutamate
    ]

    base_exchanges = [
        'EX_co2_e',
        'EX_h_e',
        'EX_h2o_e',
        'EX_nh4_e',
        'EX_o2_e',
        'EX_pi_e',       # Phosphate (essential)
    ]

    outputs = [rxn.id for rxn in model.reactions]
    #print([rxn.id for rxn in model.reactions])
    #print(len(model.reactions))

    # Set base exchange rates
    # Leave CO2, H+, Pi, O2, NH4 and H2O unconstrained
    # Constraining or varying them can cause infeasible or biologically unrealistic fluxes
    for ex in base_exchanges:
        model.reactions.get_by_id(ex).lower_bound = -default_rate

    training_data = []
    start_time = time.time()

    print(f"Generating {n_samples} FBA training samples...\n")
    sample_count = 0

    for _ in range(n_samples):
        carbon_subset = draw_subset(carbon_exchanges)

        sample = generate_training_sample(carbon_subset, outputs)
        if sample:
            training_data.append(sample)
            sample_count += 1
            if sample_count % 1000 == 0:
                print(f"Progress: {sample_count}/{n_samples}")

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    input_cols = (
        carbon_exchanges +
        base_exchanges
    )
    output_cols = [f"{rxn}_flux" for rxn in outputs]
    ordered_columns = input_cols + output_cols

    df = pd.DataFrame(training_data).reindex(columns=ordered_columns, fill_value=0.0)
                                             
    # Save data
    os.makedirs("./data", exist_ok=True)
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f"./data/{today}_full_training_data_{len(df)}_easy_samples.csv"
    df.to_csv(filename, index=False)
    
    print(f"Saved {len(df)} samples to {filename}.")

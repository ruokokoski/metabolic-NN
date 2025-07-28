import os
import numpy as np
import pandas as pd
from cobra.io import load_model
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message="Solver status is 'infeasible'")

def draw_carbon_subset(carbon_sources, selection_rate=0.5):
    n = len(carbon_sources)
    k = max(1, np.random.binomial(n, selection_rate))
    carbon_exchanges = np.array([f'EX_{met}' for met in carbon_sources])
    return np.random.choice(carbon_exchanges, size=k, replace=False)

def generate_training_sample(subset, variable_sources, outputs):
    """
    Set uptake bounds for variable subset,
    run FBA, and return the uptake rates + fluxes.
    """
    data = {}
    with model:
        # Reset variable uptakes
        for met in variable_sources:
            rxn_id = f'EX_{met}'
            if rxn_id in model.reactions:
                model.reactions.get_by_id(rxn_id).lower_bound = 0.0

        # Set subset uptakes
        for met in subset:
            # Uniform sampling:
            #rate = round(np.random.uniform(0.1, 10.0), 2) # mmol/gDW/hr
            # Sample rate log-uniformly between 0.1 and 10 mmol/gDW/hr:
            rate = round(float(10 ** np.random.uniform(-1, 1)), 4)
            model.reactions.get_by_id(met).lower_bound = -rate
            data[met] = rate

        # Add base exchange uptakes with default_rate to data dict
        for met in base_exchanges:
            data[met] = default_rate

        # Run FBA
        solution = model.optimize()
        if solution.status != 'optimal':
            return None

        for rxn_id in outputs:
            data[rxn_id + "_flux"] = solution.fluxes.get(rxn_id, 0.0)

        return data

if __name__ == "__main__":
    np.random.seed(42)
    default_rate = 100
    n_samples = 50000

    # Load the simplified E. coli metabolic model
    model = load_model("textbook")

    variable_sources = [
        'glc__D_e',   # D-Glucose
        'fru_e',      # Fructose
        'lac__D_e',   # D-Lactate
        'pyr_e',      # Pyruvate
        'ac_e',       # Acetate
        'akg_e',      # 2-Oxoglutarate
        'succ_e',     # Succinate
        'fum_e',      # Fumarate
        'mal__L_e',   # L-Malate
        'etoh_e',     # Ethanol
        'acald_e',    # Acetaldehyde
        'for_e',      # Formate
        'gln__L_e',   # L-Glutamine
        'glu__L_e'    # L-Glutamate
    ]

    base_exchanges = [
        'EX_co2_e',
        'EX_h_e',
        'EX_h2o_e',
        'EX_nh4_e',
        'EX_o2_e',
        'EX_pi_e'
    ]

    outputs = base_exchanges + ['Biomass_Ecoli_core']
    
    # Set base exhange uptakes to default_rate
    for met in base_exchanges:
        model.reactions.get_by_id(met).lower_bound = -default_rate

    training_data = []
    start_time = time.time()

    print(f"Generating {n_samples} random FBA samples with random carbon subsets...\n")
    sample_count = 0
    for _ in range(n_samples):
        subset = draw_carbon_subset(variable_sources)
        #print(f"Subset: {subset}")
        sample = generate_training_sample(subset, variable_sources, outputs)
        if sample:
            training_data.append(sample)
            sample_count += 1
            if sample_count % 1000 == 0:
                print(f"Progress: {sample_count}/{n_samples}")

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    uptake_cols = [f'EX_{m}' for m in variable_sources] + base_exchanges
    flux_cols = [f"{rxn}_flux" for rxn in outputs]
    ordered_columns = uptake_cols + flux_cols

    df = pd.DataFrame(training_data).reindex(columns=ordered_columns, fill_value=0.0)
    
    # Save data
    os.makedirs("./data", exist_ok=True)
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f"./data/{today}_training_data_{len(df)}_samples.csv"
    df.to_csv(filename, index=False)
    
    print(f"Saved {len(df)} samples to {filename}.")

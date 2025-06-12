import numpy as np
import pandas as pd
from cobra.io import load_model
import time
import warnings
warnings.filterwarnings("ignore", message="Solver status is 'infeasible'")

# Load the simplified E. coli metabolic model
model = load_model("textbook")

inputs = ['EX_glc__D_e', 'EX_o2_e', 'EX_nh4_e', 'EX_pi_e']
outputs = ['EX_co2_e', 'EX_h2o_e', 'EX_h_e', 'Biomass_Ecoli_core'] # 'EX_o2_e', 'EX_nh4_e', 'EX_pi_e',

def generate_training_sample(glc_rate, o2_rate, nh4_rate, pi_rate):
    """
    Set uptake bounds for glucose, oxygen, ammonia, phosphate,
    run FBA, and return the four uptake rates + three resulting fluxes.
    """
    with model:
        model.reactions.EX_glc__D_e.bounds = (-glc_rate, -0.1)
        model.reactions.EX_o2_e.bounds = (-o2_rate, -0.0)
        model.reactions.EX_nh4_e.bounds = (-nh4_rate, -0.1)
        model.reactions.EX_pi_e.bounds  = (-pi_rate, -0.1)

        solution = model.optimize()

        if solution.status != 'optimal':
            return None

        data = {
            "glucose_uptake": glc_rate,
            "oxygen_uptake": o2_rate,
            "ammonia_uptake":  nh4_rate,
            "phosphate_uptake": pi_rate,
        }

        for rxn_id in outputs:
            data[rxn_id + "_flux"] = solution.fluxes.get(rxn_id, 0.0)

        return data

np.random.seed(42)

training_data = []
num_samples = 20000
start_time = time.time()

print(f"Generating {num_samples} random FBA samples with 4 inputs...\n")
for i in range(num_samples):
    if i % 1000 == 0:
        print(f"Progress: {i}/{num_samples}")
    glc = round(np.random.uniform(0.1, 10.0), 2)  # mmol/gDW/hr
    o2 = round(np.random.uniform(0, 25.0), 2)
    nh4 = round(np.random.uniform(0.1, 8.0), 2)
    pi = round(np.random.uniform(0.1, 6.0), 2)

    row = generate_training_sample(glc, o2, nh4, pi)
    if row:
        training_data.append(row)

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

df = pd.DataFrame(training_data)
filename = f"./data/training_data_{len(df)}_samples.csv"
df.to_csv(filename, index=False)
print(f"Saved {len(df)}/{num_samples} successful samples to {filename}")
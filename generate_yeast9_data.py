import os
import csv
import numpy as np
import pandas as pd
from cobra.io import read_sbml_model
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message="Solver status is 'infeasible'")


def draw_subset(exchanges, max_sources=7):
    """
    Randomly draw between 1 and max_sources (default=7) exchanges.
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
            rate = random_rate(min_val=0.1, max_val=carbon_exchange_rate, log_uniform=False)
            model.reactions.get_by_id(ex).lower_bound = -rate
            data[ex] = rate

        # Set base exchange rates
        for ex in base_exchanges:
            if ex == "r_1992": # variable oxygen
                o2_rate = random_rate(min_val=1, max_val=default_rate, log_uniform=False)
                model.reactions.get_by_id("r_1992").lower_bound = -o2_rate
                data["r_1992"] = o2_rate
            elif ex == "r_1654":  # variable ammonium
                nh4_rate = random_rate(min_val=0.1, max_val=default_rate)
                model.reactions.get_by_id("r_1654").lower_bound = -nh4_rate
                data["r_1654"] = nh4_rate
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
    np.random.seed(41)

    n_samples = 250000
    default_rate = 50
    carbon_exchange_rate = 10
    batch_size = 1000

    # Load Saccharomyces Cerevisiae (yeast9) metabolic model
    model_dir ="./models"
    model = read_sbml_model(os.path.join(model_dir, "yeast-GEM.xml"))

    print("Objective reaction:", model.objective)

    carbon_exchanges = [
        # Sugars
        'r_1542', # (1->3)-beta-D-glucan
        'r_1650', # trehalose
        'r_1651', # alpha-D-glucosamine 6-phosphate
        'r_1706', # D-arabinose
        'r_1709', # D-fructose
        'r_1710', # D-galactose
        'r_1712', # D-glucitol
        'r_1714', # D-glucose
        'r_1715', # D-mannose
        'r_1716', # D-ribose
        'r_1718', # D-xylose
        'r_1808', # glycerol
        'r_1875', # L-arabinitol
        'r_1878', # L-arabinose
        'r_1931', # maltose
        'r_2058', # sucrose
        'r_2104', # xylitol
        'r_4043', # raffinose
        'r_4498', # 6-O-alpha-D-glucopyranosyl-D-fructofuranose
        'r_4499', # 5-dehydro-D-gluconate
        'r_4500', # D-tagatose
        'r_4501', # turanose
        'r_4502', # D-glucose 6-phosphate
        'r_4503', # alpha-maltotriose
        'r_4504', # D-glucose 1-phosphate
        'r_4505', # methyl alpha-D-glucopyranoside
        'r_4506', # D-Glucosamine
        'r_4507', # glycerone
        'r_4522', # glycerol 1-phosphate
        'r_4535', # glycerol 2-phosphate
        'r_4538', # 6-phospho-D-gluconate
        'r_4539', # D-mannose 6-phosphate
        'r_4547', # D-mannose 1-phosphate

        # Organic acids
        'r_1634', # acetate
        'r_1551', # (S)-lactate
        'r_2056', # succinate
        'r_1552', # (S)-malate
        'r_1798', # fumarate
        'r_1586', # 2-oxoglutarate

        # Alcohols
        'r_1761', # ethanol
        'r_1866', # isobutanol
        'r_1865', # isoamylol
        'r_4494', # methanol
        'r_1580', # 2-methylbutanol

        # Amino acids
        'r_1889', # L-glutamate
        'r_1873', # L-alanine
        'r_1881', # L-aspartate
        'r_1906', # L-serine
        'r_1899', # L-leucine
    ]

    base_exchanges = [
        'r_1654', # ammonium
        'r_1832', # H+
        'r_1861', # iron(2+)
        'r_1992', # oxygen
        'r_2005', # phosphate
        'r_2020', # potassium
        'r_2049', # sodium
        'r_2060', # sulphate
        'r_2100', # water
        'r_4593', # chloride
        'r_4594', # Cu2
        'r_4595', # Mn
        'r_4596', # Zn
        'r_4597', # Mg
        'r_4600'  # Ca
    ]

    print(f"Generating {n_samples} FBA training samples...\n")
    outputs = [rxn.id for rxn in model.reactions]
    input_cols = carbon_exchanges + base_exchanges
    output_cols = [f"{rxn}_flux" for rxn in outputs]
    ordered_columns = input_cols + output_cols

    # Prepare output file
    os.makedirs("./data", exist_ok=True)
    today = datetime.today().strftime('%Y-%m-%d')
    temp_filename = f"./data/{today}_yeast9_training_data_temp.csv"
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
    final_filename = f"./data/{today}_yeast9_data_{sample_count}_samples.csv"
    os.rename(temp_filename, final_filename)
    
    total_time = time.time() - start_time
    print(f"\nCompleted {sample_count} samples in {total_time:.2f} seconds")
    print(f"Saved to {final_filename}")
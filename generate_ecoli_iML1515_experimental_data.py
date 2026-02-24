import os
import csv
import gc
import numpy as np
import pandas as pd
from cobra.io import read_sbml_model
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message="Solver status is 'infeasible'")


def draw_subset(exchanges, max_sources=4):
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

def generate_training_sample(carbon_subset, fixed_carbon_exchanges, base_exchanges, amino_exchanges, outputs, default_rate):
    data = {}
    try:
        # Reset all exchanges
        for rxn in model.exchanges:
            rxn.lower_bound = 0.0

        # Set uptake rates for selected variable carbon sources
        for ex in carbon_subset:
            rate = random_rate(min_val=0.05, max_val=carbon_exhange_rate, log_uniform=False)

            model.reactions.get_by_id(ex).lower_bound = -rate
            data[ex] = rate

        # Set fixed carbon sources (e.g., glycerol fixed in experiments)
        for ex in fixed_carbon_exchanges:
            model.reactions.get_by_id(ex).lower_bound = -amino_rate
            data[ex] = amino_rate

        # Set base exchange rates
        for ex in base_exchanges:
            if ex == "EX_o2_e":  # variable oxygen
                o2_rate = random_rate(min_val=1, max_val=default_rate, log_uniform=False)
                model.reactions.get_by_id("EX_o2_e").lower_bound = -o2_rate
                data["EX_o2_e"] = o2_rate
            else:
                model.reactions.get_by_id(ex).lower_bound = -default_rate
                data[ex] = default_rate

        # Set amino acid exchange rates (fixed UB)
        for ex in amino_exchanges:
            model.reactions.get_by_id(ex).lower_bound = -amino_rate
            data[ex] = amino_rate

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
    np.random.seed(9) # 2, 5, 7, 8, test data: 9

    n_samples = 50000
    stitch_final_file = True
    solver_reset_interval = 5000
    default_rate = 10
    carbon_exhange_rate = 2.2 # 2.2 to match experimental set (Faure et al 2023)
    amino_rate = 2.2
    batch_size = 1000
    chunk_size = 10000

    # Load the E. coli iML1515 metabolic model
    model_dir ="./models"
    model = read_sbml_model(os.path.join(model_dir, "iML1515.xml"))

    print("Objective reaction:", model.objective)

    # Faure experimental data:
    carbon_exchanges = [
        #'EX_glc__D_e',   # D-Glucose
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
    fixed_carbon_exchanges = [
        'EX_glyc_e',     # Glycerol fixed (2.2) in Faure experimental setup
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
        'EX_o2_e', # Fixed to 2.2 in Faure setup

    # Experimental dataset contains also these:
        'EX_fe3_e', 
        'EX_sel_e', 
        'EX_tungs_e', 
        'EX_slnt_e',
    ]
    # Amino acids present in experimental dataset (Faure et al 2023).
    # These were set to 2.2:
    amino_exchanges = [
        'EX_ala__L_e', # Alanine
        'EX_pro__L_e', # Proline 
        'EX_thr__L_e', # Threonine
        'EX_gly_e',    # Glycine
    ]

    print(f"Generating {n_samples} FBA training samples...\n")
    outputs = [rxn.id for rxn in model.reactions]
    input_cols = carbon_exchanges + fixed_carbon_exchanges + base_exchanges + amino_exchanges
    output_cols = [f"{rxn}_flux" for rxn in outputs]
    ordered_columns = input_cols + output_cols

    # Prepare output files
    os.makedirs("./data", exist_ok=True)
    chunk_dir = "./data/iML1515_exp_chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    start_time = time.time()

    sample_count = 0
    chunk_count = 0
    chunk_sample_count = 0
    batch = []
    chunk_files = []

    def open_new_chunk_writer(chunk_index):
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:04d}.csv")
        fh = open(chunk_path, 'w', newline='')
        writer = csv.writer(fh)
        writer.writerow(ordered_columns)
        return fh, writer, chunk_path

    chunk_file_handle, chunk_writer, chunk_path = open_new_chunk_writer(chunk_count)
    chunk_files.append(chunk_path)

    try:
        for i in range(n_samples):
            if i > 0 and i % solver_reset_interval == 0:
                # Periodic solver/model refresh for stability in very long runs
                del model
                gc.collect()
                model = read_sbml_model(os.path.join(model_dir, "iML1515.xml"))

            carbon_subset = draw_subset(carbon_exchanges)
            
            sample = generate_training_sample(
                carbon_subset,
                fixed_carbon_exchanges,
                base_exchanges,
                amino_exchanges,
                outputs,
                default_rate
            )
            if sample:
                row = [sample.get(col, 0.0) for col in ordered_columns]
                batch.append(row)
                sample_count += 1

            # Write batch (split across chunk boundaries if needed)
            if len(batch) >= batch_size:
                pending_rows = batch
                batch = []

                while pending_rows:
                    room = chunk_size - chunk_sample_count
                    if room == 0:
                        chunk_file_handle.flush()
                        chunk_file_handle.close()
                        chunk_count += 1
                        chunk_sample_count = 0
                        chunk_file_handle, chunk_writer, chunk_path = open_new_chunk_writer(chunk_count)
                        chunk_files.append(chunk_path)
                        room = chunk_size
                    take = min(room, len(pending_rows))
                    rows_now = pending_rows[:take]
                    chunk_writer.writerows(rows_now)
                    chunk_sample_count += take
                    pending_rows = pending_rows[take:]

                    if chunk_sample_count == chunk_size and pending_rows:
                        chunk_file_handle.flush()
                        chunk_file_handle.close()
                        chunk_count += 1
                        chunk_sample_count = 0
                        chunk_file_handle, chunk_writer, chunk_path = open_new_chunk_writer(chunk_count)
                        chunk_files.append(chunk_path)

                chunk_file_handle.flush()

                elapsed = time.time() - start_time
                hours, rem = divmod(int(elapsed), 3600)
                minutes, _ = divmod(rem, 60)
                now_str = datetime.now().strftime('%H:%M')
                print(f"Generated {sample_count}/{n_samples} samples "
                    f"({hours}h {minutes}m elapsed, time {now_str}, chunk {chunk_count + 1})")

        # Write remaining batch
        if batch:
            pending_rows = batch
            while pending_rows:
                room = chunk_size - chunk_sample_count
                if room == 0:
                    chunk_file_handle.flush()
                    chunk_file_handle.close()
                    chunk_count += 1
                    chunk_sample_count = 0
                    chunk_file_handle, chunk_writer, chunk_path = open_new_chunk_writer(chunk_count)
                    chunk_files.append(chunk_path)
                    room = chunk_size
                take = min(room, len(pending_rows))
                rows_now = pending_rows[:take]
                chunk_writer.writerows(rows_now)
                chunk_sample_count += take
                pending_rows = pending_rows[take:]

                if chunk_sample_count == chunk_size and pending_rows:
                    chunk_file_handle.flush()
                    chunk_file_handle.close()
                    chunk_count += 1
                    chunk_sample_count = 0
                    chunk_file_handle, chunk_writer, chunk_path = open_new_chunk_writer(chunk_count)
                    chunk_files.append(chunk_path)

            chunk_file_handle.flush()
    finally:
        chunk_file_handle.close()

    # Save chunk manifest and optionally stitch later
    manifest_file = os.path.join(chunk_dir, "chunk_manifest.txt")
    with open(manifest_file, "w", newline="") as mf:
        for path in chunk_files:
            mf.write(path + "\n")

    final_filename = f"./data/iML1515_exp_training_data_{sample_count}_samples.csv"
    if stitch_final_file:
        # Stitch all chunk files into one final file (single header)
        with open(final_filename, 'w', newline='') as fout:
            final_writer = csv.writer(fout)
            final_writer.writerow(ordered_columns)
            for chunk_file in chunk_files:
                with open(chunk_file, 'r', newline='') as fin:
                    reader = csv.reader(fin)
                    next(reader, None)  # skip chunk header
                    final_writer.writerows(reader)

        # Remove intermediate chunk files after successful stitching
        for chunk_file in chunk_files:
            try:
                os.remove(chunk_file)
            except OSError:
                pass
        try:
            os.remove(manifest_file)
        except OSError:
            pass
        try:
            os.rmdir(chunk_dir)
        except OSError:
            pass
    
    total_time = time.time() - start_time
    total_hours, rem = divmod(int(total_time), 3600)
    total_minutes, _ = divmod(rem, 60)
    print(f"\nCompleted {sample_count} samples in {total_hours}h {total_minutes}m")
    if stitch_final_file:
        print("Intermediate chunk files removed")
        print(f"Saved to {final_filename}")
    else:
        print(f"Saved chunk files in {chunk_dir} ({len(chunk_files)} files)")
        print(f"Chunk manifest: {manifest_file}")
        print("Final stitching skipped (set stitch_final_file=True to merge).")

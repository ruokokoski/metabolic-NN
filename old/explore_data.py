import pandas as pd

datafile = "./data/2025-06-26_full_training_data_99973_samples.csv"

df = pd.read_csv(datafile)

mean = df['ACKr_flux'].mean()
std = df['ACKr_flux'].std()

print(f"ACKr_flux mean: {mean}")
print(f"ACKr_flux standard deviation: {std}")

print(df['ACKr_flux'])

def preprocess_zero_inflated(y_col, epsilon=1e-6):
    is_zero = (y_col.abs() < epsilon)
    return {
        'zero_mask': is_zero,
        'nonzero_values': y_col[~is_zero],
        'nonzero_ratio': 1 - is_zero.mean()
    }

stats = preprocess_zero_inflated(df['D_LACt2_flux'])
print(f"Non-zero ratio: {stats['nonzero_ratio']:.4f}")

'''
mean = df['EX_for_e_flux'].mean()
std = df['EX_for_e_flux'].std()

print(f"EX_for_e_flux mean: {mean}")
print(f"EX_for_e_flux standard deviation: {std}")
'''
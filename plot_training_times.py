import numpy as np
import matplotlib.pyplot as plt

training_times = np.array([32, 95, 191, 421, 511, 1511, 2301]) # Yeast9 training times in minutes (100k samples)
output_subset_ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0])

training_times_hours = training_times / 60

plt.figure(figsize=(8, 6))
plt.plot(output_subset_ratios, training_times_hours, marker='o', linewidth=2, color='blue', label='Training time (hours)')

plt.title("Training time vs Output subset ratio (100k samples)")
plt.xlabel("Output subset ratio")
plt.ylabel("Training time (hours)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
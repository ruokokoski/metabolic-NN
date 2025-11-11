import numpy as np
import matplotlib.pyplot as plt

training_times = np.array([103, 269, 505, 1294, 5906]) # Yeast9 training times in minutes (250k samples)
output_subset_ratios = np.array([0.1, 0.2, 0.3, 0.5, 1.0])

training_times_hours = training_times / 60

plt.figure(figsize=(8, 6))
plt.plot(output_subset_ratios, training_times_hours, marker='o', linewidth=2, color='blue', label='Training time (hours)')

plt.title("Training time vs Output subset ratio (100k samples)")
plt.xlabel("Output subset ratio")
plt.ylabel("Training time (hours)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
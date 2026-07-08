import os

import numpy as np
import matplotlib.pyplot as plt

training_times = np.array([103, 269, 505, 1294, 5906]) # Yeast9 training times in minutes (250k samples)
output_subset_ratios = np.array([0.1, 0.2, 0.3, 0.5, 1.0])

training_times_hours = training_times / 60

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(
    output_subset_ratios,
    training_times_hours,
    marker='o',
    linewidth=2,
    markersize=5,
    color='#2B6CB0',
)
ax.set_xlabel("Output subset ratio")
ax.set_ylabel("Training time (h)")
ax.grid(True, linestyle="--", alpha=0.35)
ax.set_xlim(0.0, 1.05)
ax.set_ylim(0, 108)
ax.set_xticks(np.arange(0.0, 1.01, 0.2))
fig.tight_layout()

out_dir = "insights/thesis"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "yeast9_subset_time.png"), dpi=300)

plt.close(fig)

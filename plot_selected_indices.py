import matplotlib.pyplot as plt

def load_histogram_txt(path):
    hist = {}
    with open(path, "r") as f:
        for line in f:
            idx, count = line.strip().split(":")
            hist[int(idx)] = int(count)
    return hist

def plot_histogram(hist, title="Selected Indices Distribution", width=1.0):
    indices = list(hist.keys())
    counts = list(hist.values())

    plt.figure(figsize=(16, 6))
    plt.bar(
        indices,
        counts,
        width=width, 
        edgecolor="none",
        linewidth=0,
    )
    plt.xlabel("Index")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    hist = load_histogram_txt("selected_indices_histogram.txt")
    plot_histogram(hist)

    hist_0_200 = {k: v for k, v in hist.items() if 0 <= k <= 200}
    plot_histogram(hist_0_200, title="Indices 0–200 Distribution", width=0.8)

    hist_800_1000 = {k: v for k, v in hist.items() if 800 <= k <= 1000}
    plot_histogram(hist_800_1000, title="Indices 800–1000 Distribution", width=0.8)

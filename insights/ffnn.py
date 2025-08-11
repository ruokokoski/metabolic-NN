import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 2))
ax.axis('off')

# Slim horizontal layout positions (spread out more)
layers = {
    "Input\n(20)": (0.1, 0.5),
    "Hidden 1\n(128)": (0.35, 0.5),
    "Hidden 2\n(128)": (0.6, 0.5),
    "Output\n(95)": (0.85, 0.5)
}

# Draw rectangle nodes
def draw_box(ax, xy, label, color):
    box_width, box_height = 0.12, 0.25
    rect = plt.Rectangle(
        (xy[0] - box_width/2, xy[1] - box_height/2),
        box_width, box_height, facecolor=color, edgecolor='black', lw=1.5
    )
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], label, ha='center', va='center', fontsize=9, fontweight='bold')

# Draw arrows
def draw_arrow(ax, start, end):
    ax.annotate(
        "", xy=end, xycoords='data', xytext=start, textcoords='data',
        arrowprops=dict(arrowstyle="->", lw=1.8, color='black')
    )

# Draw boxes
draw_box(ax, layers["Input\n(20)"], "Input\n(20)", "#ffcc99")
draw_box(ax, layers["Hidden 1\n(128)"], "Hidden 1\n(128)", "#99ccff")
draw_box(ax, layers["Hidden 2\n(128)"], "Hidden 2\n(128)", "#99ccff")
draw_box(ax, layers["Output\n(95)"], "Output\n(95)", "#ff9999")

# Draw arrows
draw_arrow(ax, (layers["Input\n(20)"][0] + 0.06, 0.5), (layers["Hidden 1\n(128)"][0] - 0.06, 0.5))
draw_arrow(ax, (layers["Hidden 1\n(128)"][0] + 0.06, 0.5), (layers["Hidden 2\n(128)"][0] - 0.06, 0.5))
draw_arrow(ax, (layers["Hidden 2\n(128)"][0] + 0.06, 0.5), (layers["Output\n(95)"][0] - 0.06, 0.5))

plt.tight_layout()
plt.show()

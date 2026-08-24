"""Generate per-level MSE ratio bar chart (Figure 2).

Data source: diagnose_option_a_2116734.out
Key visual: levels 10-14 below 1.0 (model beats baseline),
coarse levels at parity, extreme levels worse.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
})

# Verified data from diagnose_option_a_2116734.out (val set: ethanol, so2, hnnn)
levels = list(range(0, 18))
ratios = [
    1341361595.50,  # level 0 (3 samples)
    1.01,   # level 1
    1.02,   # level 2
    1.01,   # level 3
    1.01,   # level 4
    1.00,   # level 5
    1.01,   # level 6
    1.03,   # level 7
    1.10,   # level 8
    1.12,   # level 9
    0.98,   # level 10
    0.79,   # level 11
    0.67,   # level 12
    0.55,   # level 13
    0.41,   # level 14
    1.27,   # level 15
    2.83,   # level 16
    5.34,   # level 17
]
sample_counts = [3, 24, 192, 192, 192, 192, 416, 720, 912, 1216,
                 1120, 928, 1056, 784, 368, 208, 32, 32]

# Clamp for display
display_ratios = [min(r, 3.5) for r in ratios]

fig, ax = plt.subplots(figsize=(7, 3.5))

colors = []
for r in ratios:
    if r > 1.05:
        colors.append("#d32f2f")
    elif r < 0.99:
        colors.append("#388e3c")
    else:
        colors.append("#757575")

bars = ax.bar(levels, display_ratios, color=colors, edgecolor="white", linewidth=0.5)

# Reference line at 1.0
ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

# Annotate level 0 (off-chart)
ax.annotate(
    "1.3\u00d710\u2079\n(3 samples)",
    xy=(0, 3.5),
    xytext=(2.0, 3.0),
    fontsize=7,
    ha="center",
    arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=1.0),
)

# Label the green zone
ax.annotate(
    "Model beats\nbaseline",
    xy=(12, 0.4),
    fontsize=8,
    color="#388e3c",
    ha="center",
    style="italic",
)

ax.set_xlabel("Tree Level")
ax.set_ylabel("MSE Ratio (Model / Baseline)")
ax.set_xticks(levels)
ax.set_ylim(0, 3.8)
ax.set_xlim(-0.6, 17.6)

# Sample count on secondary axis
ax2 = ax.twinx()
ax2.plot(
    levels,
    [np.log10(max(s, 1)) for s in sample_counts],
    color="steelblue",
    linewidth=1.0,
    alpha=0.5,
    linestyle=":",
)
ax2.set_ylabel(r"log$_{10}$(sample count)", color="steelblue", fontsize=9)
ax2.tick_params(axis="y", labelcolor="steelblue")
ax2.set_ylim(0, 4.0)

plt.tight_layout()
plt.savefig(
    "mra_nn/paper/figures/fig2_per_level_mse.pdf", bbox_inches="tight"
)
plt.savefig(
    "mra_nn/paper/figures/fig2_per_level_mse.png", bbox_inches="tight", dpi=300
)
print("Saved fig2_per_level_mse.pdf/png")

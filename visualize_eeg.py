"""
EEG Signal Visualizer
=====================
Plots your filtered EEG channels and normalized band-power data.

RUN:
    python visualize_eeg.py

Produces four figures (also saved as PNGs in data/):
    1. Filtered EEG montage  — all 8 channels stacked
    2. Band powers for one channel (FZ) — delta/theta/alpha/beta over time
    3. Band-power heatmap — every channel x band
    4. Theta/Beta ratio — the attention metric that drives the game
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FILTERED_CSV = os.path.join(DATA_DIR, "Filtered_EEG.csv")
BANDPOWER_CSV = os.path.join(DATA_DIR, "Normalized_BandPower.csv")

# dark style to match the game
plt.rcParams.update({
    "figure.facecolor": "#0a0e1c",
    "axes.facecolor": "#0f1524",
    "axes.edgecolor": "#3a4468",
    "axes.labelcolor": "#c8d0e0",
    "xtick.color": "#8a94b4",
    "ytick.color": "#8a94b4",
    "text.color": "#e6e6f0",
    "grid.color": "#1e2440",
    "font.size": 10,
})

BAND_COLORS = {
    "Delta": "#5a8ad4",
    "Theta": "#d4b45a",
    "Alpha": "#5ad49a",
    "Beta":  "#d45a7a",
}


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader]
    return header, np.array(rows)


# ── load ──
f_hdr, f_data = load_csv(FILTERED_CSV)
b_hdr, b_data = load_csv(BANDPOWER_CSV)

f_time = f_data[:, 0] - f_data[0, 0]      # start at 0
b_time = b_data[:, 0] - b_data[0, 0]
channels = f_hdr[1:]                        # FP1, FP2, ...


# ══════════════════════════════════════════════
#  FIG 1 — Filtered EEG montage (stacked channels)
# ══════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(12, 8))
offset_step = 180                            # vertical spacing between channels
palette = plt.cm.cool(np.linspace(0.2, 0.9, len(channels)))

for i, ch in enumerate(channels):
    offset = (len(channels) - i) * offset_step
    ax1.plot(f_time, f_data[:, i + 1] + offset, color=palette[i], linewidth=0.9)
    ax1.text(-0.02, offset, ch, ha="right", va="center",
             color=palette[i], fontweight="bold", fontsize=11)

ax1.set_title("Filtered EEG — Channel Montage", fontsize=14, fontweight="bold", pad=15)
ax1.set_xlabel("Time (s)")
ax1.set_yticks([])
ax1.margins(x=0.02)
ax1.grid(True, axis="x", alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(DATA_DIR, "fig1_filtered_montage.png"), dpi=130)


# ══════════════════════════════════════════════
#  FIG 2 — Band powers for FZ (frontal midline)
# ══════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(12, 5))
target_ch = "FZ"
for band, color in BAND_COLORS.items():
    col = f"{target_ch}_{band}_Relative_Z"
    if col in b_hdr:
        idx = b_hdr.index(col)
        ax2.plot(b_time, b_data[:, idx], color=color, linewidth=1.6, label=band)

ax2.axhline(0, color="#3a4468", linewidth=0.8, linestyle="--")
ax2.set_title(f"{target_ch} Band Powers (Z-scored, Relative)",
              fontsize=14, fontweight="bold", pad=15)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Relative power (Z)")
ax2.legend(loc="upper right", framealpha=0.2, facecolor="#0f1524", edgecolor="#3a4468")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(DATA_DIR, "fig2_fz_bandpowers.png"), dpi=130)


# ══════════════════════════════════════════════
#  FIG 3 — Band-power heatmap (channel x band, averaged over time)
# ══════════════════════════════════════════════
eeg_channels = ["FP1", "FP2", "O1", "O2", "C3", "CZ", "FZ", "C4"]
bands = ["Delta", "Theta", "Alpha", "Beta"]
heat = np.zeros((len(eeg_channels), len(bands)))
for ci, ch in enumerate(eeg_channels):
    for bi, band in enumerate(bands):
        col = f"{ch}_{band}_Relative_Z"
        if col in b_hdr:
            heat[ci, bi] = np.mean(b_data[:, b_hdr.index(col)])

fig3, ax3 = plt.subplots(figsize=(7, 8))
im = ax3.imshow(heat, aspect="auto", cmap="RdYlBu_r", vmin=-1, vmax=1)
ax3.set_xticks(range(len(bands)))
ax3.set_xticklabels(bands)
ax3.set_yticks(range(len(eeg_channels)))
ax3.set_yticklabels(eeg_channels)
ax3.set_title("Mean Band Power by Channel\n(Z-scored)", fontsize=13,
              fontweight="bold", pad=15)
for ci in range(len(eeg_channels)):
    for bi in range(len(bands)):
        ax3.text(bi, ci, f"{heat[ci, bi]:.2f}", ha="center", va="center",
                 color="#111", fontsize=9, fontweight="bold")
cbar = fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label("Relative power (Z)")
fig3.tight_layout()
fig3.savefig(os.path.join(DATA_DIR, "fig3_bandpower_heatmap.png"), dpi=130)


# ══════════════════════════════════════════════
#  FIG 4 — Theta/Beta attention metric (drives the game)
# ══════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(12, 5))
for ch, color in [("FZ", "#d45a7a"), ("CZ", "#5ad49a")]:
    theta = b_data[:, b_hdr.index(f"{ch}_Theta_Relative_Z")]
    beta = b_data[:, b_hdr.index(f"{ch}_Beta_Relative_Z")]
    focus_metric = beta - theta            # higher = more focused
    ax4.plot(b_time, focus_metric, color=color, linewidth=1.8, label=f"{ch} (β−θ)")

ax4.axhline(0, color="#3a4468", linewidth=0.8, linestyle="--")
ax4.fill_between(b_time, 0, 4, color="#5ad49a", alpha=0.06)
ax4.text(b_time[-1] * 0.02, 3.2, "more focused", color="#5ad49a", fontsize=9)
ax4.text(b_time[-1] * 0.02, -3.4, "less focused", color="#d45a7a", fontsize=9)
ax4.set_title("Attention Metric (Beta − Theta) — what drives the rocket",
              fontsize=14, fontweight="bold", pad=15)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Beta − Theta (Z)")
ax4.legend(loc="upper right", framealpha=0.2, facecolor="#0f1524", edgecolor="#3a4468")
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig(os.path.join(DATA_DIR, "fig4_attention_metric.png"), dpi=130)

print("Saved 4 figures to data/:")
print("  fig1_filtered_montage.png")
print("  fig2_fz_bandpowers.png")
print("  fig3_bandpower_heatmap.png")
print("  fig4_attention_metric.png")

plt.show()   # opens interactive windows

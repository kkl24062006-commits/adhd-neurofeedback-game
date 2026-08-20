"""
EEG FFT + Band-Power Bar Plots
==============================
Two views:
    1. FFT amplitude spectrum for each EEG channel (with band regions shaded)
    2. Band-power bar charts (grouped per channel, and averaged across channels)

RUN:
    python fft_bandpower_plots.py

Saves PNGs into data/.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FILTERED_CSV = os.path.join(DATA_DIR, "Filtered_EEG.csv")
BANDPOWER_CSV = os.path.join(DATA_DIR, "Normalized_BandPower.csv")

# ── EEG bands (Hz) ──
BANDS = {
    "Delta": (0.5, 4,  "#5a8ad4"),
    "Theta": (4,   8,  "#d4b45a"),
    "Alpha": (8,   13, "#5ad49a"),
    "Beta":  (13,  30, "#d45a7a"),
}

# dark style
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


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader]
    return header, np.array(rows)


# ══════════════════════════════════════════════
#  FIGURE 1 — FFT amplitude spectrum per channel
# ══════════════════════════════════════════════
f_hdr, f_data = load_csv(FILTERED_CSV)
channels = f_hdr[1:]
t = f_data[:, 0]
fs = (len(t) - 1) / (t[-1] - t[0])      # estimated sampling rate
n = len(t)
freqs = np.fft.rfftfreq(n, d=1.0 / fs)
FMAX = 40                                # only show EEG-relevant range

fig1, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
axes = axes.flatten()

for i, ch in enumerate(channels):
    ax = axes[i]
    sig = f_data[:, i + 1]
    sig = sig - np.mean(sig)              # remove DC
    windowed = sig * np.hanning(n)        # reduce spectral leakage
    amp = np.abs(np.fft.rfft(windowed)) * 2 / n

    # shade the band regions
    for band, (lo, hi, color) in BANDS.items():
        ax.axvspan(lo, hi, color=color, alpha=0.12)

    ax.plot(freqs, amp, color="#5ad4d4", linewidth=1.2)
    ax.set_xlim(0, FMAX)
    ax.set_title(ch, fontsize=12, fontweight="bold")
    if i >= 4:
        ax.set_xlabel("Frequency (Hz)")
    if i % 4 == 0:
        ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

# one shared legend for the band colours
handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4)
           for _, _, c in BANDS.values()]
fig1.legend(handles, list(BANDS.keys()), loc="upper center",
            ncol=4, framealpha=0.2, facecolor="#0f1524", edgecolor="#3a4468")
fig1.suptitle(f"FFT Amplitude Spectrum  (fs ≈ {fs:.0f} Hz)",
              fontsize=15, fontweight="bold", y=1.02)
fig1.tight_layout()
fig1.savefig(os.path.join(DATA_DIR, "fft_spectrum.png"), dpi=130,
             bbox_inches="tight")


# ══════════════════════════════════════════════
#  FIGURE 2 — Band-power bar chart (grouped per channel)
# ══════════════════════════════════════════════
b_hdr, b_data = load_csv(BANDPOWER_CSV)
eeg_channels = ["FP1", "FP2", "O1", "O2", "C3", "CZ", "FZ", "C4"]
band_names = list(BANDS.keys())

# mean Z-scored power per channel per band
means = np.zeros((len(eeg_channels), len(band_names)))
for ci, ch in enumerate(eeg_channels):
    for bi, band in enumerate(band_names):
        col = f"{ch}_{band}_Relative_Z"
        if col in b_hdr:
            means[ci, bi] = np.mean(b_data[:, b_hdr.index(col)])

fig2, ax2 = plt.subplots(figsize=(14, 6))
x = np.arange(len(eeg_channels))
bw = 0.2
for bi, band in enumerate(band_names):
    color = BANDS[band][2]
    ax2.bar(x + (bi - 1.5) * bw, means[:, bi], bw,
            label=band, color=color, edgecolor="#0a0e1c", linewidth=0.5)

ax2.axhline(0, color="#3a4468", linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(eeg_channels)
ax2.set_ylabel("Mean relative power (Z)")
ax2.set_xlabel("Channel")
ax2.set_title("Band Power by Channel", fontsize=15, fontweight="bold", pad=15)
ax2.legend(framealpha=0.2, facecolor="#0f1524", edgecolor="#3a4468")
ax2.grid(True, axis="y", alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(DATA_DIR, "bandpower_bars_by_channel.png"), dpi=130)


# ══════════════════════════════════════════════
#  FIGURE 3 — Band-power bar chart (averaged across all channels)
# ══════════════════════════════════════════════
band_avg = means.mean(axis=0)            # average over channels
fig3, ax3 = plt.subplots(figsize=(8, 6))
colors = [BANDS[b][2] for b in band_names]
bars = ax3.bar(band_names, band_avg, color=colors,
               edgecolor="#0a0e1c", linewidth=1)
ax3.axhline(0, color="#3a4468", linewidth=0.8)
for bar, val in zip(bars, band_avg):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             val + (0.02 if val >= 0 else -0.05),
             f"{val:.2f}", ha="center",
             va="bottom" if val >= 0 else "top",
             fontweight="bold", fontsize=11)
ax3.set_ylabel("Mean relative power (Z)")
ax3.set_title("Average Band Power (all channels)",
              fontsize=15, fontweight="bold", pad=15)
ax3.grid(True, axis="y", alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(DATA_DIR, "bandpower_bars_average.png"), dpi=130)

print("Saved 3 figures to data/:")
print("  fft_spectrum.png")
print("  bandpower_bars_by_channel.png")
print("  bandpower_bars_average.png")

plt.show()

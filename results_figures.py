"""
Results Figures — Clean Academic Style
=======================================
    python results_figures.py
Reads CSVs from data/, saves PNGs there.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "text.color": "#222222",
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
})


def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader]
    return header, np.array(rows)


def load_dict_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


# ══════════════════════════════════════════════
#  FIGURE 1 — Power Spectral Density: Raw vs Filtered (FZ)
# ══════════════════════════════════════════════

def figure1():
    raw_hdr, raw_data = load_csv("Raw_EEG.csv")
    filt_hdr, filt_data = load_csv("Filtered_EEG.csv")

    raw_fz = raw_data[:, raw_hdr.index("FZ")]
    filt_fz = filt_data[:, filt_hdr.index("FZ")]

    # estimate sampling rates from timestamps
    raw_t = raw_data[:, 0]
    filt_t = filt_data[:, 0]
    raw_fs = (len(raw_t) - 1) / (raw_t[-1] - raw_t[0])
    filt_fs = (len(filt_t) - 1) / (filt_t[-1] - filt_t[0])

    # remove DC
    raw_fz = raw_fz - np.mean(raw_fz)
    filt_fz = filt_fz - np.mean(filt_fz)

    # compute PSD
    f_raw, psd_raw = welch(raw_fz, fs=raw_fs,
                           nperseg=min(len(raw_fz), int(raw_fs * 2)))
    f_filt, psd_filt = welch(filt_fz, fs=filt_fs,
                             nperseg=min(len(filt_fz), int(filt_fs * 2)))

    # convert to dB
    psd_raw_db = 10 * np.log10(psd_raw + 1e-20)
    psd_filt_db = 10 * np.log10(psd_filt + 1e-20)

    fmax = 45  # show up to 45 Hz

    # band shading definitions
    bands = [
        ("Delta\n0.5–4 Hz", 0.5, 4, "#3498db"),
        ("Theta\n4–8 Hz", 4, 8, "#f39c12"),
        ("Alpha\n8–13 Hz", 8, 13, "#27ae60"),
        ("Beta\n13–30 Hz", 13, 30, "#e74c3c"),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Raw PSD
    for label, lo, hi, color in bands:
        ax1.axvspan(lo, hi, color=color, alpha=0.08)
    ax1.plot(f_raw[f_raw <= fmax], psd_raw_db[f_raw <= fmax],
             color="#2c5f8a", linewidth=0.9)
    ax1.set_ylabel("Power spectral density (dB)")
    ax1.set_title("(a) Raw EEG — Channel FZ", loc="left", fontweight="bold")
    ax1.set_xlim(0, fmax)
    ax1.grid(True, alpha=0.4)

    # Filtered PSD
    for label, lo, hi, color in bands:
        ax2.axvspan(lo, hi, color=color, alpha=0.08)
        mid = (lo + hi) / 2
        ax2.text(mid, ax2.get_ylim()[0] if ax2.get_ylim()[0] != 0 else -10,
                 "", ha="center", fontsize=7, color=color)
    ax2.plot(f_filt[f_filt <= fmax], psd_filt_db[f_filt <= fmax],
             color="#c0392b", linewidth=0.9)
    ax2.set_ylabel("Power spectral density (dB)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_title("(b) Filtered EEG — Channel FZ (0.5–30 Hz bandpass applied)",
                  loc="left", fontweight="bold")
    ax2.set_xlim(0, fmax)
    ax2.grid(True, alpha=0.4)

    # add band labels to top of figure via ax1
    for label, lo, hi, color in bands:
        mid = (lo + hi) / 2
        ax1.text(mid, ax1.get_ylim()[1] * 0.95, label.split("\n")[0],
                 ha="center", fontsize=8, color=color, fontweight="bold",
                 va="top")

    # mark filter edges on filtered plot
    ax2.axvline(0.5, color="#7f8c8d", linestyle="--", linewidth=0.7)
    ax2.axvline(30, color="#7f8c8d", linestyle="--", linewidth=0.7)
    ax2.text(30.5, ax2.get_ylim()[1] * 0.9 if ax2.get_ylim()[1] != 0 else 0,
             "30 Hz\ncutoff", fontsize=8, color="#7f8c8d", va="top")

    fig.tight_layout(h_pad=2.0)
    fig.savefig(os.path.join(DATA_DIR, "fig1_psd_raw_vs_filtered.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    print("  fig1_psd_raw_vs_filtered.png")


# ══════════════════════════════════════════════
#  FIGURE 2 — TBR trajectory + Theta vs Beta
# ══════════════════════════════════════════════

def figure2():
    rows, header = load_dict_csv("EEG_BandPower.csv")

    timestamps = np.array([float(r["Timestamp"]) for r in rows])
    timestamps -= timestamps[0]

    theta_fz = np.array([float(r["FZ_Theta"]) for r in rows])
    beta_fz = np.array([float(r["FZ_Beta"]) for r in rows])
    theta_cz = np.array([float(r["CZ_Theta"]) for r in rows])
    beta_cz = np.array([float(r["CZ_Beta"]) for r in rows])

    theta = (theta_fz + theta_cz) / 2
    beta = (beta_fz + beta_cz) / 2
    tbr = theta / (beta + 1e-9)

    # smooth
    k = np.ones(5) / 5
    tbr_s = np.convolve(tbr, k, mode="same")
    theta_s = np.convolve(theta, k, mode="same")
    beta_s = np.convolve(beta, k, mode="same")

    try:
        sess, _ = load_dict_csv("session_log.csv")
        baseline = float(sess[0]["mean_tbr"])
    except Exception:
        baseline = np.median(tbr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # (a) TBR
    ax1.plot(timestamps, tbr_s, color="#2c5f8a", linewidth=0.9, label="TBR")
    z = np.polyfit(np.arange(len(tbr_s)), tbr_s, 1)
    ax1.plot(timestamps, np.polyval(z, np.arange(len(tbr_s))),
             "--", color="#e67e22", linewidth=1.8, label="Linear trend")
    ax1.axhline(baseline, color="#7f8c8d", linestyle=":", linewidth=1.0,
                label=f"Session baseline ({baseline:.2f})")
    ax1.set_ylabel("Theta/Beta ratio (unitless)")
    ax1.set_title("(a) Theta/Beta ratio over time — lower indicates stronger attention",
                  loc="left", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.4)

    # (b) Theta vs Beta
    ax2.plot(timestamps, theta_s, color="#f39c12", linewidth=1.0,
             label="Theta (4–8 Hz)")
    ax2.plot(timestamps, beta_s, color="#e74c3c", linewidth=1.0,
             label="Beta (13–30 Hz)")
    ax2.set_ylabel("Band power (µV²/Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("(b) Theta and Beta band power over time (FZ + CZ average)",
                  loc="left", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.4)

    fig.tight_layout(h_pad=1.5)
    fig.savefig(os.path.join(DATA_DIR, "fig2_tbr_theta_beta.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    print("  fig2_tbr_theta_beta.png")


# ══════════════════════════════════════════════
#  FIGURE 3 — Band Power 1st vs 2nd Half
# ══════════════════════════════════════════════

def figure3():
    rows, header = load_dict_csv("EEG_BandPower.csv")
    mid = len(rows) // 2
    bands = ["Delta", "Theta", "Alpha", "Beta"]
    freq_ranges = ["0.5–4", "4–8", "8–13", "13–30"]
    bar_colors = ["#3498db", "#f39c12", "#27ae60", "#e74c3c"]

    h1_vals, h2_vals, pct_vals = [], [], []
    for band in bands:
        vals = np.mean([
            np.array([float(r[f"{ch}_{band}"]) for r in rows])
            for ch in ["FZ", "CZ"]
        ], axis=0)
        h1 = float(np.mean(vals[:mid]))
        h2 = float(np.mean(vals[mid:]))
        pct = ((h2 - h1) / (abs(h1) + 1e-9)) * 100
        h1_vals.append(h1)
        h2_vals.append(h2)
        pct_vals.append(pct)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bands))
    w = 0.3

    ax.bar(x - w / 2, h1_vals, w, label="1st half (beginning)",
           color="#bdc3c7", edgecolor="#7f8c8d", linewidth=0.6)
    ax.bar(x + w / 2, h2_vals, w, label="2nd half (end)",
           color=bar_colors, edgecolor="#555555", linewidth=0.6)

    for i, pct in enumerate(pct_vals):
        y = max(h1_vals[i], h2_vals[i]) * 1.06
        sign = "+" if pct > 0 else ""
        good = (bands[i] == "Theta" and pct < 0) or \
               (bands[i] in ("Beta", "Alpha") and pct > 0)
        ax.text(x[i], y, f"{sign}{pct:.1f}%",
                ha="center", fontweight="bold", fontsize=10,
                color="#27ae60" if good else "#c0392b")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n({r} Hz)" for b, r in zip(bands, freq_ranges)],
                       fontsize=10)
    ax.set_ylabel("Mean absolute band power (µV²/Hz)")
    ax.set_title("Band power change within session (FZ + CZ average)",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "fig3_band_power_halves.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    print("  fig3_band_power_halves.png")


# ══════════════════════════════════════════════
#  FIGURE 4 — Cross-Session Comparison
# ══════════════════════════════════════════════

def figure4():
    sess, _ = load_dict_csv("session_log.csv")
    n = len(sess)
    x = np.arange(1, n + 1)
    tbrs = [float(s["mean_tbr"]) for s in sess]
    scores = [int(s["final_score"]) for s in sess]
    levels = [s["session_level"] for s in sess]
    durs = [float(s["duration_sec"]) for s in sess]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # (a) TBR
    ax1.bar(x, tbrs, width=0.45, color="#2c5f8a", edgecolor="#1a3a5c", linewidth=0.6)
    for i in range(n):
        ax1.text(x[i], tbrs[i] + 0.015, f"{tbrs[i]:.3f}",
                 ha="center", fontsize=10, fontweight="bold")
        ax1.text(x[i], tbrs[i] / 2, f"Level {levels[i]}",
                 ha="center", fontsize=9, color="white", fontweight="bold")
    if n >= 2:
        z = np.polyfit(x, tbrs, 1)
        ax1.plot(x, np.polyval(z, x), "--", color="#e67e22", linewidth=1.8)
    ax1.set_xlabel("Session")
    ax1.set_ylabel("Mean TBR (unitless)")
    ax1.set_title("(a) Mean TBR per session", loc="left", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"S{i}" for i in x])
    ax1.grid(True, axis="y", alpha=0.4)
    ax1.set_axisbelow(True)

    # (b) Score
    ax2.bar(x, scores, width=0.45, color="#27ae60", edgecolor="#1a6b3c", linewidth=0.6)
    for i in range(n):
        ax2.text(x[i], scores[i] + 10, str(scores[i]),
                 ha="center", fontsize=10, fontweight="bold")
        ax2.text(x[i], scores[i] / 2, f"{durs[i]:.0f} s",
                 ha="center", fontsize=9, color="white", fontweight="bold")
    ax2.set_xlabel("Session")
    ax2.set_ylabel("Game score (points)")
    ax2.set_title("(b) Game score per session", loc="left", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S{i}" for i in x])
    ax2.grid(True, axis="y", alpha=0.4)
    ax2.set_axisbelow(True)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(DATA_DIR, "fig4_cross_session.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    print("  fig4_cross_session.png")


# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures...\n")
    figure1()
    figure2()
    figure3()
    figure4()
    print(f"\nSaved to {DATA_DIR}/")
    plt.show()

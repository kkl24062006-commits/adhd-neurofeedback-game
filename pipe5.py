# ============================================================
# REAL-TIME EEG + PPG PREPROCESSING + EEG BANDPOWER PIPELINE
# ============================================================

# FEATURES INCLUDED
# ------------------------------------------------------------
# EEG:
#   ✔ Raw EEG Saving
#   ✔ Baseline Correction
#   ✔ Bandpass Filtering (0.5–30 Hz)
#   ✔ Artifact Rejection
#   ✔ Savitzky-Golay Smoothing
#   ✔ Delta Bandpower
#   ✔ Theta Bandpower
#   ✔ Alpha Bandpower
#   ✔ Beta Bandpower
#   ✔ Bandpower CSV Saving
#
# PPG:
#   ✔ Raw PPG Saving
#   ✔ Smoothing
#   ✔ Peak Detection
#   ✔ BPM Calculation
#   ✔ RR Interval Extraction
#   ✔ RMSSD
#   ✔ SDNN
#
# ============================================================

from pylsl import StreamInlet, resolve_stream
from collections import deque

import numpy as np
import time
import csv
import os

from scipy.signal import (
    butter,
    filtfilt,
    savgol_filter,
    find_peaks,
    welch
)

# ============================================================
# DISPLAY SAVE LOCATION
# ============================================================

print("\nSaving CSV files at:")
print(os.getcwd())

# ============================================================
# SETTINGS
# ============================================================

EEG_STREAM = 'NEW1_EEG'
PPG_STREAM = 'NEW1_PPG'

EEG_FS = 250
PPG_FS = 100

WINDOW_SECONDS = 4

# ============================================================
# CHANNEL NAMES
# ============================================================

CHANNEL_NAMES = [
    'FP1',
    'FP2',
    'O1',
    'O2',
    'C3',
    'CZ',
    'FZ',
    'C4'
]

# ============================================================
# BUFFER SIZES
# ============================================================

EEG_BUFFER_SIZE = EEG_FS * WINDOW_SECONDS
PPG_BUFFER_SIZE = PPG_FS * WINDOW_SECONDS

# ============================================================
# CONNECT EEG STREAM
# ============================================================

print("🔌 Connecting EEG stream...")

eeg_streams = resolve_stream('name', EEG_STREAM)

if len(eeg_streams) == 0:
    raise RuntimeError("❌ EEG stream not found")

eeg_inlet = StreamInlet(eeg_streams[0])

print("✅ EEG connected")

# ============================================================
# CONNECT PPG STREAM
# ============================================================

print("🔌 Connecting PPG stream...")

ppg_streams = resolve_stream('name', PPG_STREAM)

if len(ppg_streams) == 0:
    raise RuntimeError("❌ PPG stream not found")

ppg_inlet = StreamInlet(ppg_streams[0])

print("✅ PPG connected")

# ============================================================
# CREATE BUFFERS
# ============================================================

eeg_buffer = deque(maxlen=EEG_BUFFER_SIZE)
ppg_buffer = deque(maxlen=PPG_BUFFER_SIZE)

# ============================================================
# CREATE CSV FILES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

raw_eeg_file = open(
    os.path.join(DATA_DIR, "Raw_EEG.csv"),
    "w",
    newline=''
)

filtered_eeg_file = open(
    os.path.join(DATA_DIR, "Filtered_EEG.csv"),
    "w",
    newline=''
)

ppg_file = open(
    os.path.join(DATA_DIR, "PPG_Data.csv"),
    "w",
    newline=''
)

bandpower_file = open(
    os.path.join(DATA_DIR, "EEG_BandPower.csv"),
    "w",
    newline=''
)

# ============================================================
# CSV WRITERS
# ============================================================

raw_eeg_writer = csv.writer(raw_eeg_file)
filtered_eeg_writer = csv.writer(filtered_eeg_file)
ppg_writer = csv.writer(ppg_file)
bandpower_writer = csv.writer(bandpower_file)

# ============================================================
# WRITE HEADERS
# ============================================================

raw_eeg_writer.writerow(
    ["Timestamp"] + CHANNEL_NAMES
)

filtered_eeg_writer.writerow(
    ["Timestamp"] + CHANNEL_NAMES
)

ppg_writer.writerow(
    ["Timestamp", "PPG"]
)

bandpower_header = ["Timestamp"]

for ch in CHANNEL_NAMES:

    bandpower_header.extend([
        f"{ch}_Delta",
        f"{ch}_Theta",
        f"{ch}_Alpha",
        f"{ch}_Beta"
    ])

bandpower_writer.writerow(bandpower_header)

# ============================================================
# EEG BANDPASS FILTER
# ============================================================

lowcut = 0.5
highcut = 30

b, a = butter(
    4,
    [lowcut / (EEG_FS / 2), highcut / (EEG_FS / 2)],
    btype='band'
)

# ============================================================
# ARTIFACT REJECTION FUNCTION
# ============================================================

def artifact_rejection(signal, threshold=3):

    mean = np.mean(signal)
    std = np.std(signal)

    upper = mean + threshold * std
    lower = mean - threshold * std

    cleaned = np.clip(signal, lower, upper)

    return cleaned

# ============================================================
# HRV FEATURE FUNCTION
# ============================================================

def compute_hrv(rr_intervals):

    if len(rr_intervals) < 2:
        return 0, 0

    rr_diff = np.diff(rr_intervals)

    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    sdnn = np.std(rr_intervals)

    return rmssd, sdnn

# ============================================================
# EEG BANDPOWER FUNCTION
# ============================================================

def bandpower(data, sf, band):

    low, high = band

    freqs, psd = welch(
        data,
        sf,
        nperseg=sf * 2
    )

    freq_res = freqs[1] - freqs[0]

    idx_band = np.logical_and(
        freqs >= low,
        freqs <= high
    )

    bp = np.sum(psd[idx_band]) * freq_res

    return bp

# ============================================================
# REAL-TIME LOOP
# ============================================================

print("\n📡 REAL-TIME PREPROCESSING STARTED")
print("Press Ctrl+C to stop\n")

try:

    while True:

        # =====================================================
        # GET EEG SAMPLE
        # =====================================================

        eeg_sample, eeg_time = eeg_inlet.pull_sample()

        eeg_buffer.append(eeg_sample)

        # =====================================================
        # SAVE RAW EEG SAMPLE
        # =====================================================

        raw_eeg_writer.writerow(
            [eeg_time] + list(eeg_sample)
        )

        raw_eeg_file.flush()

        # =====================================================
        # GET PPG SAMPLE
        # =====================================================

        ppg_sample, ppg_time = ppg_inlet.pull_sample()

        ppg_buffer.append(ppg_sample)

        # =====================================================
        # SAVE RAW PPG SAMPLE
        # =====================================================

        ppg_writer.writerow(
            [ppg_time] + list(ppg_sample)
        )

        ppg_file.flush()

        # =====================================================
        # WAIT UNTIL BUFFERS FILL
        # =====================================================

        if (
            len(eeg_buffer) < EEG_BUFFER_SIZE or
            len(ppg_buffer) < PPG_BUFFER_SIZE
        ):

            continue

        # =====================================================
        # CONVERT TO NUMPY ARRAYS
        # =====================================================

        eeg_data = np.array(eeg_buffer)
        ppg_data = np.array(ppg_buffer)

        # =====================================================
        # EEG PREPROCESSING
        # =====================================================

        # -----------------------------------------------------
        # BASELINE CORRECTION
        # -----------------------------------------------------

        eeg_data = eeg_data - np.mean(
            eeg_data,
            axis=0
        )

        # -----------------------------------------------------
        # BANDPASS FILTERING
        # -----------------------------------------------------

        eeg_filtered = filtfilt(
            b,
            a,
            eeg_data,
            axis=0
        )

        # -----------------------------------------------------
        # ARTIFACT REJECTION
        # -----------------------------------------------------

        for ch in range(eeg_filtered.shape[1]):

            eeg_filtered[:, ch] = artifact_rejection(
                eeg_filtered[:, ch]
            )

        # -----------------------------------------------------
        # SAVITZKY-GOLAY SMOOTHING
        # -----------------------------------------------------

        for ch in range(eeg_filtered.shape[1]):

            eeg_filtered[:, ch] = savgol_filter(
                eeg_filtered[:, ch],
                window_length=11,
                polyorder=2
            )

        # =====================================================
        # EEG BAND SEPARATION
        # =====================================================

        delta_band = (0.5, 4)
        theta_band = (4, 8)
        alpha_band = (8, 13)
        beta_band = (13, 30)

        delta_powers = []
        theta_powers = []
        alpha_powers = []
        beta_powers = []

        for ch in range(eeg_filtered.shape[1]):

            signal = eeg_filtered[:, ch]

            delta_power = bandpower(
                signal,
                EEG_FS,
                delta_band
            )

            theta_power = bandpower(
                signal,
                EEG_FS,
                theta_band
            )

            alpha_power = bandpower(
                signal,
                EEG_FS,
                alpha_band
            )

            beta_power = bandpower(
                signal,
                EEG_FS,
                beta_band
            )

            delta_powers.append(delta_power)
            theta_powers.append(theta_power)
            alpha_powers.append(alpha_power)
            beta_powers.append(beta_power)

        # =====================================================
        # SAVE EEG BANDPOWERS
        # =====================================================

        bandpower_row = [eeg_time]

        for i in range(len(CHANNEL_NAMES)):

            bandpower_row.extend([

                delta_powers[i],
                theta_powers[i],
                alpha_powers[i],
                beta_powers[i]

            ])

        bandpower_writer.writerow(bandpower_row)

        bandpower_file.flush()

        # =====================================================
        # SAVE FILTERED EEG
        # =====================================================

        latest_filtered = eeg_filtered[-1]

        filtered_eeg_writer.writerow(
            [eeg_time] + list(latest_filtered)
        )

        filtered_eeg_file.flush()

        # =====================================================
        # PPG PREPROCESSING
        # =====================================================

        ppg_signal = ppg_data[:, 0]

        # -----------------------------------------------------
        # PPG SMOOTHING
        # -----------------------------------------------------

        ppg_smoothed = savgol_filter(
            ppg_signal,
            window_length=11,
            polyorder=2
        )

        # -----------------------------------------------------
        # PEAK DETECTION
        # -----------------------------------------------------

        peaks, _ = find_peaks(
            ppg_smoothed,
            distance=PPG_FS * 0.5,
            prominence=np.std(ppg_smoothed) * 0.5
        )

        # =====================================================
        # HRV CALCULATIONS
        # =====================================================

        if len(peaks) > 1:

            rr_intervals = np.diff(peaks) / PPG_FS

            bpm = 60 / np.mean(rr_intervals)

            rmssd, sdnn = compute_hrv(rr_intervals)

        else:

            bpm = 0
            rmssd = 0
            sdnn = 0

        # =====================================================
        # DISPLAY OUTPUT
        # =====================================================

        print("\n================================================")
        print("✅ PREPROCESSING COMPLETED")
        print("================================================")

        print("\n🧠 EEG INFORMATION")
        print("--------------------------------")

        print(f"EEG Shape : {eeg_filtered.shape}")

        print("\nLatest Smoothed EEG Values")
        print(np.round(eeg_filtered[-1], 3))

        # =====================================================
        # DISPLAY EEG BAND POWERS
        # =====================================================

        print("\n🧠 EEG BAND POWERS")
        print("--------------------------------")

        for i, ch in enumerate(CHANNEL_NAMES):

            print(f"\n{ch}")

            print(f"Delta : {delta_powers[i]:.4f}")
            print(f"Theta : {theta_powers[i]:.4f}")
            print(f"Alpha : {alpha_powers[i]:.4f}")
            print(f"Beta  : {beta_powers[i]:.4f}")

        # =====================================================
        # CHANNEL MAPPING
        # =====================================================

        print("\nChannel Mapping")

        for i, ch in enumerate(CHANNEL_NAMES):

            print(f"{i} → {ch}")

        # =====================================================
        # PPG / HRV OUTPUT
        # =====================================================

        print("\n❤️ PPG / HRV INFORMATION")
        print("--------------------------------")

        print(f"BPM    : {bpm:.2f}")
        print(f"RMSSD  : {rmssd:.4f}")
        print(f"SDNN   : {sdnn:.4f}")
        print(f"Peaks  : {len(peaks)}")

        # =====================================================
        # WINDOW STATUS
        # =====================================================

        print("\n🟢 Window processed successfully")

        time.sleep(0.5)

# ============================================================
# STOP CONDITION
# ============================================================

except KeyboardInterrupt:

    print("\n\n🛑 Real-time preprocessing stopped")

    raw_eeg_file.close()
    filtered_eeg_file.close()
    ppg_file.close()
    bandpower_file.close()

    print("✅ CSV files saved successfully")
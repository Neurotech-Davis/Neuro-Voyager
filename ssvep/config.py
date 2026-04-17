"""Shared constants for SSVEP pipeline."""

# Target frequencies (Hz)
TARGET_FREQS = [7.0, 9.0, 13.0, 17.0]

# Sampling rate (OpenBCI Cyton default)
SFREQ = 250.0

# Preprocessing
BANDPASS = (5.0, 35.0)  # Hz — upper at 35 to preserve 2nd harmonics up to 34 Hz
NOTCH = 60.0             # Hz, power-line

# CCA
N_HARMONICS = 3
WINDOW_SEC = 2.0
STEP_SEC = 0.5
CCA_THRESHOLD = 0.3

# LSL stream types
EEG_STREAM_TYPE = "EEG"
MARKER_STREAM_TYPE = "Markers"

# Epoch extraction
ONSET_SKIP_SEC = 0.5  # skip transient at stimulus onset

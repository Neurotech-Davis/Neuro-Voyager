"""Single source of truth for all SSVEP pipeline constants."""

# OpenBCI Cyton pin → 10-20 channel names
CHANNEL_MAPPING = {
    "EEG_1": "O10",   # Pin 1, Grey
    "EEG_2": "O9",    # Pin 2, Purple
    "EEG_3": "O2",    # Pin 3, Blue
    "EEG_4": "O1",    # Pin 4, Green
    "EEG_5": "PO4",   # Pin 5, Yellow
    "EEG_6": "PO3",   # Pin 6, Orange
    "EEG_7": "Pz",    # Pin 7, Red
    "EEG_8": "Oz",    # Pin 8, Brown
}
MONTAGE_CHANNELS = ["O2", "O1", "PO4", "PO3", "Pz", "Oz"]
NON_STANDARD_CHANNELS = ["O9", "O10"]

# Preprocessing parameters
BANDPASS = (1.0, 50.0)
NOTCH = 60.0
ICA_MAX_EXCLUDE = 2
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = 800
KURT_THRESHOLD = 2.5

# Epoch parameters
EPOCH_DUR = 5.0          # default stimulus duration (used when marker has no duration)
TARGET_FREQS = [7, 9, 13, 15]  # Hz, expected stimulus frequencies

# Marker detection
MARKER_PREFIX = "stim_"  # normalized prefix applied to all marker descriptions

# Stream types recognized as marker streams (case-insensitive).
# "Markers" is the XDF/LSL standard; "Tags" is sometimes used in practice.
MARKER_STREAM_TYPES = {"markers", "tags"}

# Regex patterns to extract a frequency (Hz) from a marker label.
# Tried in order; first match wins. Each must have a capture group for the number.
# Covers: "stim_7", "freq_13", "7hz", "7 Hz", "7", etc.
MARKER_PATTERNS = [
    r"^stim[_\s]?(\d+(?:\.\d+)?)\s*(?:hz)?$",   # stim_7, stim7, stim_7hz, stim 7
    r"^freq[_\s]?(\d+(?:\.\d+)?)\s*(?:hz)?$",    # freq_7, freq7, freq_7hz
    r"^(\d+(?:\.\d+)?)\s*hz$",                    # 7hz, 7 Hz, 13.5hz
    r"^(\d+(?:\.\d+)?)$",                         # 7, 13, 15 (bare number)
]

# Analysis parameters
WELCH_NPERSEG_SEC = 2.0  # Welch segment length in seconds
SNR_NOISE_BAND = (1.0, 4.0)  # Hz distance from target for noise estimation

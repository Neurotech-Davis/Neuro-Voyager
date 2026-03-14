"""Single source of truth for experiment timing and analysis constants."""

# --- Experiment Timing ---
TARGET_FREQS = [7, 9, 13, 17]  # Hz, stimulus frequencies (presentation order)
STIM_ON_SEC = 5.0  # seconds stimulus is active
BREAK_SEC = 2.0  # seconds break between frequencies
LONG_BREAK_SEC = 10.0  # seconds break after each outer loop
INNER_ITERATIONS = 5  # repetitions of the 4-freq set per outer loop
OUTER_ITERATIONS = 3  # number of outer loops

# Derived timing (do not edit)
FREQ_PERIOD_SEC = STIM_ON_SEC + BREAK_SEC  # 7s
INNER_CYCLE_SEC = len(TARGET_FREQS) * FREQ_PERIOD_SEC  # 28s
INNER_LOOP_SEC = INNER_ITERATIONS * INNER_CYCLE_SEC  # 140s
OUTER_LOOP_SEC = INNER_LOOP_SEC + LONG_BREAK_SEC  # 150s
TOTAL_EXPERIMENT_SEC = OUTER_ITERATIONS * OUTER_LOOP_SEC  # 450s

# --- Preprocessing ---
BANDPASS = (5.0, 35.0)  # Hz, bandpass filter range for SSVEP
NOTCH = 60.0  # Hz, power-line notch

# --- CCA Parameters ---
NHARMONICS = 3  # Number of harmonics to include in reference signals
WINDOW_SEC = 2.0  # Length of the analysis window in seconds
STEP_SEC = 0.5    # How often to run classification

# --- SSVEP Targets & Commands ---
# Frequency (Hz) -> Command Character
TARGETS = {
    7.0:  'F',  # Forward
    9.0:  'L',  # Left
    11.0: 'R',  # Right
    13.0: 'B',  # Backward
    0.0:  'S'   # Stop/Idle
}

# --- Serial Communication ---
SERIAL_PORT = '/dev/ttys001'  # Default for testing, change as needed
SERIAL_BAUD = 9600
# For local simulation, we might use a pair of virtual ports
SIM_PORT_A = '/dev/ttys003'
SIM_PORT_B = '/dev/ttys004'

# --- PSD / Analysis ---
WELCH_NPERSEG_SEC = 2.0  # Welch segment length in seconds
SNR_NOISE_BAND = (1.0, 4.0)  # Hz distance from target for noise estimation

import numpy as np
import matplotlib.pyplot as plt
import pyxdf

# -------------------------------
# Load XDF
# -------------------------------
def load_xdf(file_path):
    streams, _ = pyxdf.load_xdf(file_path)

    eeg_stream = None
    marker_stream = None

    for s in streams:
        if s['info']['type'][0] == 'EEG':
            eeg_stream = s
        elif s['info']['type'][0] == 'Markers':
            marker_stream = s

    if eeg_stream is None or marker_stream is None:
        raise ValueError("EEG or Marker stream not found")

    signal = np.array(eeg_stream['time_series']).T  # shape: (channels, samples)
    timestamps = np.array(eeg_stream['time_stamps'])

    marker_times = np.array(marker_stream['time_stamps'])
    marker_values = marker_stream['time_series']

    sfreq = float(eeg_stream['info']['nominal_srate'][0])

    return signal, timestamps, marker_times, marker_values, sfreq


# -------------------------------
# Extract epochs (FIXED LENGTH)
# -------------------------------
def extract_epochs(signal, timestamps, marker_times, tmin, tmax, sfreq):
    epochs = []
    expected_len = int((tmax - tmin) * sfreq)

    for mt in marker_times:
        start_time = mt + tmin
        end_time = mt + tmax

        start_idx = np.searchsorted(timestamps, start_time)
        end_idx = start_idx + expected_len

        if end_idx <= signal.shape[1]:
            epoch = signal[:, start_idx:end_idx]

            if epoch.shape[1] == expected_len:
                epochs.append(epoch)

    return np.stack(epochs)  # safe now


# -------------------------------
# Plot raw EEG
# -------------------------------
def plot_raw(signal, sfreq):
    t = np.arange(signal.shape[1]) / sfreq

    plt.figure(figsize=(12, 6))
    for i in range(signal.shape[0]):
        plt.plot(t, signal[i] + i * 100, label=f'Ch {i+1}')  # offset for visibility

    plt.xlabel("Time (s)")
    plt.title("Raw EEG Signals")
    plt.show()


# -------------------------------
# FFT (frequency domain)
# -------------------------------
def compute_fft(epoch, sfreq):
    n = epoch.shape[1]
    freqs = np.fft.rfftfreq(n, d=1/sfreq)

    fft_vals = np.abs(np.fft.rfft(epoch, axis=1))
    return freqs, fft_vals


def plot_fft(freqs, fft_vals):
    plt.figure(figsize=(10, 5))

    for ch in range(fft_vals.shape[0]):
        plt.plot(freqs, fft_vals[ch], alpha=0.5)

    plt.xlim(0, 50)  # SSVEP range
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT per Channel")
    plt.show()


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=2.0)

    args = parser.parse_args()

    print(f"Loading: {args.file}")

    signal, timestamps, marker_times, marker_values, sfreq = load_xdf(args.file)

    print(f"Signal shape: {signal.shape}")
    print(f"Sampling rate: {sfreq} Hz")

    # Raw visualization
    plot_raw(signal, sfreq)

    # Epoching
    epochs = extract_epochs(signal, timestamps, marker_times, args.tmin, args.tmax, sfreq)

    print(f"Epochs shape: {epochs.shape}")  # (n_trials, channels, samples)

    # Take average across trials
    avg_epoch = np.mean(epochs, axis=0)

    # FFT
    freqs, fft_vals = compute_fft(avg_epoch, sfreq)

    plot_fft(freqs, fft_vals)
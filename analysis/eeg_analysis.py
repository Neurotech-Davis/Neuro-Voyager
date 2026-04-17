import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


# RemoveBaselineDrift
def remove_drift(signal, fs):
    # Option 1: detrend (simple, robust)
    signal = detrend(signal)

    # Option 2: high-pass filter (better for EEG)
    nyq = 0.5 * fs
    b, a = butter(2, 0.5 / nyq, btype='high')  # 0.5 Hz cutoff
    signal = filtfilt(b, a, signal)

    return signal


# ExtractEpochsUsingMarkers
def extract_epochs(data, timestamps, marker_times, tmin=0.0, tmax=5.0):
    epochs = []
    fs = 1 / np.mean(np.diff(timestamps))

    for t in marker_times:
        start_idx = np.searchsorted(timestamps, t + tmin)
        end_idx = np.searchsorted(timestamps, t + tmax)

        if end_idx < len(data):
            epochs.append(data[start_idx:end_idx])

    return np.array(epochs)  # shape: (n_epochs, samples)


# AverageEpochs
def average_epochs(epochs):
    return np.mean(epochs, axis=0)


# ComputeFFT
def compute_fft(signal, fs):
    N = len(signal)
    freqs = fftfreq(N, 1/fs)
    fft_vals = np.abs(fft(signal)) / N

    mask = freqs > 0
    return freqs[mask], fft_vals[mask]


# PlotFFT
def plot_fft(freqs, fft_vals, title="FFT"):
    plt.figure(figsize=(8,4))
    plt.plot(freqs, fft_vals)
    plt.xlim(0, 40)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.show()
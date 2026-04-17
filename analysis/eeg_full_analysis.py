import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import fft, fftfreq


# ParseCommandLineArguments
parser = argparse.ArgumentParser(description="Full EEG Analysis Pipeline")
parser.add_argument("--file", type=str, required=True, help="Path to .xdf file")
parser.add_argument("--channel", type=int, default=0, help="Channel index (0-based)")
parser.add_argument("--tmin", type=float, default=0.0, help="Epoch start (seconds)")
parser.add_argument("--tmax", type=float, default=5.0, help="Epoch end (seconds)")
parser.add_argument("--filter", action="store_true", help="Apply bandpass filter")
args = parser.parse_args()


# LoadXDFFile
print("Loading:", args.file)
streams, _ = pyxdf.load_xdf(args.file)


# FindEEGStream
eeg_stream = None
for s in streams:
    if s["info"]["type"][0].lower() == "eeg":
        eeg_stream = s
        break

if eeg_stream is None:
    raise RuntimeError("No EEG stream found")


# FindMarkerStream
marker_stream = None
for s in streams:
    if s["info"]["type"][0].lower() == "markers":
        marker_stream = s
        break

if marker_stream is None:
    raise RuntimeError("No marker stream found")


# ExtractEEGData
data = np.array(eeg_stream["time_series"])
timestamps = np.array(eeg_stream["time_stamps"])
fs = float(eeg_stream["info"]["nominal_srate"][0])

n_channels = data.shape[1]

if args.channel < 0 or args.channel >= n_channels:
    raise ValueError(f"Channel must be between 0 and {n_channels-1}")

signal = data[:, args.channel] * 1e6  # ConvertToMicrovolts


# DriftCorrection
def remove_drift(sig, fs):
    sig = detrend(sig)
    nyq = 0.5 * fs
    b, a = butter(2, 0.5 / nyq, btype='high')
    return filtfilt(b, a, sig)

signal = remove_drift(signal, fs)


# OptionalBandpassFilter
def bandpass(sig, fs, low=5, high=40):
    nyq = 0.5 * fs
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

if args.filter:
    signal = bandpass(signal, fs)


# ExtractMarkers
marker_times = np.array(marker_stream["time_stamps"])


# EpochExtraction
def extract_epochs(signal, timestamps, marker_times, tmin, tmax):
    epochs = []

    for t in marker_times:
        start_idx = np.searchsorted(timestamps, t + tmin)
        end_idx = np.searchsorted(timestamps, t + tmax)

        if end_idx < len(signal):
            epochs.append(signal[start_idx:end_idx])

    return np.array(epochs)


epochs = extract_epochs(signal, timestamps, marker_times, args.tmin, args.tmax)

if len(epochs) == 0:
    raise RuntimeError("No valid epochs extracted")

print("Extracted epochs:", epochs.shape)


# EpochAveraging
avg_epoch = np.mean(epochs, axis=0)


# FFTComputation
def compute_fft(sig, fs):
    N = len(sig)
    freqs = fftfreq(N, 1/fs)
    fft_vals = np.abs(fft(sig)) / N

    mask = freqs > 0
    return freqs[mask], fft_vals[mask]

freqs, fft_vals = compute_fft(avg_epoch, fs)


# PeakDetection
peak_freq = freqs[np.argmax(fft_vals)]
print("Detected peak frequency:", peak_freq)


# TimeAxis
time = timestamps - timestamps[0]
epoch_time = np.linspace(args.tmin, args.tmax, len(avg_epoch))


# PlotFullSignal
plt.figure(figsize=(10, 4))
plt.plot(time, signal)
plt.title("Drift-Corrected Signal (Full)")
plt.xlabel("Time (s)")
plt.ylabel("µV")
plt.grid(True)


# PlotAverageEpoch
plt.figure(figsize=(10, 4))
plt.plot(epoch_time, avg_epoch)
plt.title("Average Epoch")
plt.xlabel("Time (s)")
plt.ylabel("µV")
plt.grid(True)


# PlotFFT
plt.figure(figsize=(10, 4))
plt.plot(freqs, fft_vals)
plt.xlim(0, 40)
plt.axvline(peak_freq, color='r', linestyle='--', label=f"Peak: {peak_freq:.2f} Hz")
plt.title("FFT of Averaged Epoch")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.show()
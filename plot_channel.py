import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.signal import butter, filtfilt

# ParseCommandLineArguments
parser = argparse.ArgumentParser(description="Plot EEG channels from XDF")
parser.add_argument("--file", type=str, required=True, help="Path to .xdf file")
parser.add_argument("--channel", type=int, default=None, help="Channel index (0-based)")
parser.add_argument("--all", action="store_true", help="Plot all channels and save PNGs")
parser.add_argument("--full", action="store_true", help="Plot full duration")
parser.add_argument("--duration", type=float, default=5.0, help="Seconds to plot")
parser.add_argument("--filter", action="store_true", help="Apply bandpass filter (5–40 Hz)")
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

# ExtractEEGData
data = np.array(eeg_stream["time_series"])
timestamps = np.array(eeg_stream["time_stamps"])
fs = float(eeg_stream["info"]["nominal_srate"][0])

n_channels = data.shape[1]

# CreateOutputDirectory
base_name = os.path.splitext(os.path.basename(args.file))[0]
output_dir = os.path.join("output", base_name)
os.makedirs(output_dir, exist_ok=True)

# DefineBandpassFilter
def bandpass(signal, fs, low=5, high=40):
    nyq = 0.5 * fs
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

# CreateTimeAxis
time = timestamps - timestamps[0]
if args.full:
    samples = len(time)
else:
    samples = int(args.duration * fs)

# PlotAndSaveFunction
def plot_channel(ch_idx):
    signal = data[:, ch_idx] * 1e6  # ConvertToMicrovolts

    if args.filter:
        signal = bandpass(signal, fs)

    plt.figure(figsize=(10, 5))
    plt.plot(time[:samples], signal[:samples], linewidth=1)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"{base_name} - Channel {ch_idx}")

    plt.grid(True)

    save_path = os.path.join(output_dir, f"channel_{ch_idx}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")

# RunSingleOrAllChannels
if args.all:
    for ch in range(n_channels):
        plot_channel(ch)
else:
    if args.channel is None:
        raise ValueError("Provide --channel or use --all")

    if args.channel < 0 or args.channel >= n_channels:
        raise ValueError(f"Channel must be between 0 and {n_channels-1}")

    plot_channel(args.channel)
"""Minimal EEG viewer: load, filter, visualize."""

import argparse
import numpy as np
import mne
import pyxdf
mne.set_log_level("WARNING")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def load_eeg_data(file_path):
    #Step 4: Load data into MN
    streams, file_header = pyxdf.load_xdf(file_path)

    # Automate finding the EEG stream since streams[0] is often a Marker stream
    try:
        eeg_stream = next(stream for stream in streams if stream["info"]["type"][0].lower() == "eeg")
    except StopIteration:
        # Fallback to the stream with the largest time_series if type is missing
        eeg_stream = max(streams, key=lambda stream: len(stream["time_series"]))

    # Extract data (channels x time) and sampling frequency
    eeg_data_array = np.array(eeg_stream["time_series"]).T          # transpose so channels first
    sampling_frequency = float(eeg_stream["info"]["nominal_srate"][0])
    
    # Create channel names (fallback if not present)
    number_of_channels = eeg_data_array.shape[0]
    channel_names = [f"EEG {i+1}" for i in range(number_of_channels - 1)] + ["STI 014"]  # last = stim/marker if present

    # Create info object (all channels as EEG except last as stim if applicable)
    channel_types = ['eeg'] * (number_of_channels - 1) + ['stim']   # adjust based on your file

    # Create the MNE info object (basically assigning the data some metadata for MNE's reference)
    mne_info = mne.create_info(ch_names=channel_names,
                        sfreq=sampling_frequency,
                        ch_types=channel_types)

    # Create the Raw object
    raw_eeg_data = mne.io.RawArray(eeg_data_array, mne_info, verbose=False)

    # Optional: Add annotations from marker stream if you have one
    # if len(streams) > 1 and streams[1]["info"]["type"][0] == "Markers":
    #     markers = streams[1]
    #     # convert marker times & descriptions - MNE Annotations
    #     # (see MNE docs or MNELAB examples for full code)

    print(raw_eeg_data.info)
    print(f"Channels: {raw_eeg_data.ch_names}")
    print(f"Duration: {raw_eeg_data.times[-1]:.1f} seconds")
        
    return raw_eeg_data


def filter_eeg_data(raw_eeg_data):
    raw_eeg_data.load_data()
    raw_eeg_data.filter(l_freq=1.0, h_freq=25.0, fir_design="firwin")
    raw_eeg_data.notch_filter(freqs=60.0)
    return raw_eeg_data


def plot_eeg_data(eeg_data_object, plot_title, window_milliseconds=5000.0):
    # Plot raw data
    eeg_data_object.plot(
        duration=10,               # initial time window (seconds)
        n_channels=5,              # how many channels to show initially
        scalings='auto',           # auto vertical scaling per channel
        title=f"{plot_title} - Time Series",
        show_scrollbars=True,      # try to show them
        show_scalebars=True,       # shows microV / unit scale bars
        block=False                # Don't block yet, so we can show PSD too
    )

    # Step 7: Plot the Power Spectral Density (PSD)
    print("Computing and plotting Power Spectral Density (PSD)...")
    psd_figure = eeg_data_object.compute_psd(fmax=30).plot(show=False)
    psd_figure.canvas.manager.set_window_title(f"{plot_title} - PSD")

    # Add vertical lines at target frequencies
    target_freqs = [7, 9, 13, 17]
    for ax in psd_figure.axes:
        for freq in target_freqs:
            ax.axvline(x=freq, color='r', linestyle='--', alpha=0.5)

    # Block execution here to keep all windows open
    plt.show(block=True)
    # figure.savefig('raw_plot.png')
    # Do the same for filtered.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG viewer")
    parser.add_argument("filepath", help="Path to .xdf or .fif file")
    parser.add_argument("--raw", action="store_true", help="Skip filtering")
    parsed_arguments = parser.parse_args()

    import os
    experiment_name = os.path.splitext(os.path.basename(parsed_arguments.filepath))[0]

    eeg_data = load_eeg_data(parsed_arguments.filepath)
    if not parsed_arguments.raw:
        eeg_data = filter_eeg_data(eeg_data)
    plot_eeg_data(eeg_data, experiment_name)

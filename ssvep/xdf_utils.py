"""XDF loading and data utilities."""

import csv
import numpy as np
import mne
import pyxdf

from . import config

mne.set_log_level("WARNING")


def load_xdf(filepath):
    """Load XDF file, return separated EEG and marker streams.

    Returns:
        dict with keys:
            eeg_stream: the EEG stream dict (or None)
            marker_stream: the marker stream dict (or None)
            header: XDF file header
    """
    streams, header = pyxdf.load_xdf(filepath)

    eeg_stream = None
    marker_stream = None

    for s in streams:
        stype = s["info"]["type"][0].lower()
        if stype == config.EEG_STREAM_TYPE.lower() and eeg_stream is None:
            eeg_stream = s
        elif stype == config.MARKER_STREAM_TYPE.lower() and marker_stream is None:
            marker_stream = s

    if eeg_stream is None:
        # Fallback: largest stream by sample count
        eeg_stream = max(streams, key=lambda s: len(s["time_series"]))

    return {"eeg_stream": eeg_stream, "marker_stream": marker_stream, "header": header}


def xdf_to_mne(filepath):
    """Load XDF and convert to MNE Raw + marker list.

    Returns:
        (mne.io.Raw, markers)
        markers: list of (timestamp, label) tuples, or None if no marker stream.
    """
    result = load_xdf(filepath)
    eeg = result["eeg_stream"]

    data = eeg["time_series"].T  # (n_channels, n_samples)
    sfreq = float(eeg["info"]["nominal_srate"][0])

    # Scale uV → V if needed
    if np.max(np.abs(data)) > 1.0:
        data = data * 1e-6

    n_channels = data.shape[0]
    ch_names = [f"Ch{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    # Extract markers
    markers = None
    ms = result["marker_stream"]
    if ms is not None:
        timestamps = ms["time_stamps"]
        labels = [str(row[0]) if len(row) > 0 else "" for row in ms["time_series"]]
        markers = list(zip(timestamps, labels))

    return raw, markers


def load_realtime_log(csv_path):
    """Read a CSV log produced by real-time mode.

    Returns:
        list of dicts with keys: timestamp, detected_freq, correlation, latency_ms
    """
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp": float(row["timestamp"]),
                "detected_freq": float(row["detected_freq"]) if row["detected_freq"] else None,
                "correlation": float(row["correlation"]),
                "latency_ms": float(row["latency_ms"]),
            })
    return rows

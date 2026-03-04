"""Load raw EEG data and clean it (bandpass, notch, CAR, ICA)."""

import re

import mne
import numpy as np

from config import (
    BANDPASS, CHANNEL_MAPPING, EPOCH_DUR, ICA_MAX_EXCLUDE, ICA_MAX_ITER,
    ICA_RANDOM_STATE, KURT_THRESHOLD, MARKER_PATTERNS, MARKER_PREFIX,
    MARKER_STREAM_TYPES, NOTCH,
)

# Compile marker patterns once
_MARKER_RES = [re.compile(p, re.IGNORECASE) for p in MARKER_PATTERNS]


def _parse_marker_freq(label):
    """Try to extract a frequency (Hz) from a marker label string.

    Returns the frequency as a float, or None if no pattern matches.
    """
    for pat in _MARKER_RES:
        m = pat.match(label)
        if m:
            return float(m.group(1))
    return None


def _extract_markers(streams, eeg_stream):
    """Extract stimulus markers from XDF marker streams.

    Scans all streams whose type is in MARKER_STREAM_TYPES for labels
    that match any of MARKER_PATTERNS. Returns list of
    (onset_sec, duration, description) tuples with times relative to
    the EEG stream start.
    """
    eeg_t0 = eeg_stream["time_stamps"][0]

    markers = []
    for stream in streams:
        if stream["info"]["type"][0].lower() not in MARKER_STREAM_TYPES:
            continue
        timestamps = stream["time_stamps"]
        if timestamps is None or len(timestamps) == 0:
            continue
        for ts, vals in zip(timestamps, stream["time_series"]):
            label = vals[0] if isinstance(vals, list) else str(vals)
            freq = _parse_marker_freq(label.strip())
            if freq is not None:
                onset = ts - eeg_t0
                freq_str = str(int(freq)) if freq == int(freq) else str(freq)
                markers.append((onset, EPOCH_DUR, f"{MARKER_PREFIX}{freq_str}"))

    markers.sort(key=lambda m: m[0])
    return markers


def load_raw(filepath: str) -> mne.io.Raw:
    """Load EEG from XDF or FIF into an MNE Raw object.

    For XDF: extracts the EEG stream, converts uV->V, renames channels
    using CHANNEL_MAPPING, sets a standard 10-20 montage, and attaches
    stimulus markers as MNE Annotations (persisted in FIF).
    For FIF: loads directly (annotations already embedded).
    """
    if filepath.endswith(".xdf"):
        import pyxdf
        streams, _ = pyxdf.load_xdf(filepath)
        eeg_stream = next(
            s for s in streams if s["info"]["type"][0].lower() == "eeg"
        )

        data = eeg_stream["time_series"].T * 1e-6  # uV -> V
        srate = float(eeg_stream["info"]["nominal_srate"][0])
        n_channels = int(eeg_stream["info"]["channel_count"][0])

        # Extract channel names from stream descriptor, fallback to EEG_1..N
        try:
            desc_channels = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
            ch_names = [ch["label"][0] for ch in desc_channels]
        except (KeyError, IndexError, TypeError):
            ch_names = [f"EEG_{i+1}" for i in range(n_channels)]

        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Rename channels to 10-20 names
        rename = {k: v for k, v in CHANNEL_MAPPING.items() if k in raw.ch_names}
        if rename:
            raw.rename_channels(rename)

        # Set standard montage (O9/O10 are non-standard, so warn)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="warn")

        # Extract stimulus markers and attach as annotations
        markers = _extract_markers(streams, eeg_stream)
        if markers:
            onsets, durations, descriptions = zip(*markers)
            annotations = mne.Annotations(
                onset=list(onsets),
                duration=list(durations),
                description=list(descriptions),
            )
            raw.set_annotations(annotations)
            print(f"  Found {len(markers)} stimulus markers")
        else:
            print("  WARNING: No stimulus markers found in XDF.")
            print("    Expected: 'stim_7', 'freq_13', '7hz', '15', etc.")

    elif filepath.endswith(".fif"):
        raw = mne.io.read_raw_fif(filepath, preload=True)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    return raw


def clean(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply bandpass, notch, CAR, and ICA artifact removal.

    Returns the cleaned Raw object (modified in-place).
    """
    raw.load_data()

    # Bandpass filter
    raw.filter(l_freq=BANDPASS[0], h_freq=BANDPASS[1], fir_design="firwin")

    # Notch filter (power line)
    raw.notch_filter(freqs=NOTCH)

    # Common Average Reference
    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()

    # ICA artifact removal (rank drops by 1 after CAR)
    ica = mne.preprocessing.ICA(
        n_components=len(raw.ch_names) - 1,
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
    )
    ica.fit(raw)

    bads = set()

    # 1) Muscle artifacts
    try:
        muscle_idx, _ = ica.find_bads_muscle(raw)
        bads.update(muscle_idx)
    except Exception:
        pass

    # 2) Kurtosis-based detection
    sources = ica.get_sources(raw).get_data()
    mean = sources.mean(axis=1, keepdims=True)
    std = sources.std(axis=1, keepdims=True) + 1e-12
    kurt = np.mean(((sources - mean) / std) ** 4, axis=1)
    z_kurt = (kurt - kurt.mean()) / (kurt.std() + 1e-12)
    bads.update(np.where(np.abs(z_kurt) > KURT_THRESHOLD)[0].tolist())

    # 3) EOG proxy (Pz is the best available frontal-ish channel)
    try:
        eog_idx, _ = ica.find_bads_eog(raw, ch_name="Pz")
        bads.update(eog_idx)
    except Exception:
        pass

    # Cap at ICA_MAX_EXCLUDE, keeping worst by kurtosis z-score
    bads = sorted(bads,
                  key=lambda i: abs(z_kurt[i]) if i < len(z_kurt) else 0,
                  reverse=True)[:ICA_MAX_EXCLUDE]

    ica.exclude = bads
    ica.apply(raw)

    return raw

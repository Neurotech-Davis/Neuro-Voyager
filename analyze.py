"""Epoch segmentation, PSD/SNR computation, and result plotting."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.signal import welch

from config import EPOCH_DUR, MARKER_PREFIX, SNR_NOISE_BAND, WELCH_NPERSEG_SEC

# Match annotations normalized by preprocess.py (e.g. "stim_7", "stim_13.5")
_MARKER_RE = re.compile(
    rf"^{re.escape(MARKER_PREFIX)}(\d+(?:\.\d+)?)$", re.IGNORECASE
)


def segment_epochs(raw):
    """Segment continuous data into stimulus epochs using MNE Annotations.

    Reads annotations matching MARKER_PREFIX (e.g. "stim_7") from the raw
    object. Each annotation defines an epoch with onset, duration, and
    target frequency.

    Returns list of dicts with keys: s0, s1, target_freq, time.
    """
    sfreq = raw.info["sfreq"]
    n_samples = raw.n_times

    epochs = []
    for ann in raw.annotations:
        m = _MARKER_RE.match(ann["description"])
        if not m:
            continue

        freq = float(m.group(1))
        target_f = int(freq) if freq == int(freq) else freq
        onset = ann["onset"]
        duration = ann["duration"] if ann["duration"] > 0 else EPOCH_DUR

        s0 = int(onset * sfreq)
        s1 = int((onset + duration) * sfreq)
        if s0 < 0 or s1 > n_samples:
            continue
        epochs.append({"s0": s0, "s1": s1, "target_freq": target_f, "time": onset})

    epochs.sort(key=lambda e: e["time"])

    if not epochs:
        print(f"  WARNING: No epochs found. Ensure annotations match "
              f"'{MARKER_PREFIX}<freq>' (e.g. '{MARKER_PREFIX}7').")

    return epochs


def compute_snr(psd_vals, freqs, target_hz):
    """Compute SNR in dB at target_hz relative to surrounding noise band.

    Returns (snr_db, signal_power).
    """
    idx = np.argmin(np.abs(freqs - target_hz))
    nb_lo, nb_hi = SNR_NOISE_BAND
    dist = np.abs(freqs - target_hz)
    neighbors = (dist > nb_lo) & (dist <= nb_hi)
    if neighbors.sum() == 0:
        return float("nan"), psd_vals[idx]
    noise = psd_vals[neighbors].mean()
    snr = 10 * np.log10(psd_vals[idx] / noise) if noise > 0 else 0.0
    return snr, psd_vals[idx]


def compute_psd_per_target(data, epochs, sfreq, ch_names):
    """Compute Welch PSD per epoch, then average by target frequency.

    Returns dict keyed by target_freq, each containing:
        freqs: frequency axis
        avg_psd: (n_channels, n_freqs) array averaged over epochs
        snr_f0: (n_channels,) array of SNR at fundamental
        snr_h2: (n_channels,) array of SNR at 2nd harmonic
    """
    nperseg = int(WELCH_NPERSEG_SEC * sfreq)
    n_channels = len(ch_names)

    # Group epochs by target frequency
    freq_buckets = {}
    for ep in epochs:
        freq_buckets.setdefault(ep["target_freq"], []).append(ep)

    results = {}
    for tf in sorted(freq_buckets):
        all_psds = []
        for ep in freq_buckets[tf]:
            seg = data[:, ep["s0"]:ep["s1"]]
            seg_nperseg = min(nperseg, seg.shape[1])
            freqs, psd = welch(seg, fs=sfreq, nperseg=seg_nperseg,
                               noverlap=seg_nperseg // 2, axis=1)
            all_psds.append(psd)

        avg_psd = np.mean(all_psds, axis=0)  # (n_channels, n_freqs)

        # SNR at fundamental and 2nd harmonic per channel
        snr_f0 = np.array([compute_snr(avg_psd[ci], freqs, tf)[0]
                           for ci in range(n_channels)])
        snr_h2 = np.array([compute_snr(avg_psd[ci], freqs, tf * 2)[0]
                           for ci in range(n_channels)])

        results[tf] = {
            "freqs": freqs,
            "avg_psd": avg_psd,
            "snr_f0": snr_f0,
            "snr_h2": snr_h2,
        }

    return results


def _plot_spectrum_grid(freq_data, ch_names, trial_name, output_dir,
                        transform_fn, ylabel, title_suffix, filename_suffix):
    """Shared helper for PSD and amplitude spectrum grid plots.

    Args:
        freq_data: dict from compute_psd_per_target.
        ch_names: list of channel names.
        trial_name: used in title and filename.
        output_dir: directory for saved PNG.
        transform_fn: callable(avg_psd_masked, freqs_masked) -> y_values.
        ylabel: y-axis label string.
        title_suffix: appended to trial_name in suptitle.
        filename_suffix: appended to trial_name in filename.
    """
    targets = sorted(freq_data.keys())
    n = len(targets)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(f"{trial_name} — {title_suffix}", fontsize=14)

    for idx, tf in enumerate(targets):
        ax = axes[idx // ncols][idx % ncols]
        d = freq_data[tf]
        freqs = d["freqs"]
        mask = (freqs >= 1) & (freqs <= 50)

        y_vals = transform_fn(d["avg_psd"][:, mask], freqs[mask])
        for ci, ch in enumerate(ch_names):
            ax.plot(freqs[mask], y_vals[ci], label=ch, linewidth=0.9)

        ax.axvline(tf, color="red", linestyle="--", alpha=0.7, label=f"F0={tf} Hz")
        ax.axvline(tf * 2, color="orange", linestyle="--", alpha=0.5, label=f"H2={tf*2} Hz")
        ax.set_title(f"Target {tf} Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{trial_name}_{filename_suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_results(freq_data, ch_names, trial_name, output_dir):
    """Generate PSD (dB) and amplitude spectrum (uV) plots.

    Saves two PNGs to output_dir:
        {trial_name}_psd_per_freq.png
        {trial_name}_amplitude_spectrum.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if not freq_data:
        print("No frequency data to plot.")
        return

    def to_psd_db(psd, freqs):
        return 10 * np.log10(psd + 1e-30)

    def to_amplitude_uv(psd, freqs):
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        return np.sqrt(psd * df) * 1e6  # V -> uV

    _plot_spectrum_grid(freq_data, ch_names, trial_name, output_dir,
                        to_psd_db, "Power (dB)",
                        "Epoch-Averaged PSD (dB)", "psd_per_freq")

    _plot_spectrum_grid(freq_data, ch_names, trial_name, output_dir,
                        to_amplitude_uv, "Amplitude (uV)",
                        "Amplitude Spectrum (uV)", "amplitude_spectrum")


def plot_time_series(raw, trial_name, window_ms=5000.0):
    """Interactive scrollable time-series plot of all channels in uV.

    Args:
        raw: MNE Raw object.
        trial_name: Used in the window title.
        window_ms: Visible window width in milliseconds.
    """
    data = raw.get_data() * 1e6  # V -> uV
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    n_channels, n_samples = data.shape
    total_ms = (n_samples / sfreq) * 1000.0
    times_ms = np.arange(n_samples) / sfreq * 1000.0

    # Vertical offsets so channels don't overlap
    offsets = np.zeros(n_channels)
    for i in range(1, n_channels):
        # Separate by the range of the previous channel + padding
        offsets[i] = offsets[i - 1] - (np.ptp(data[i - 1]) + np.ptp(data[i])) * 0.6

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.15)
    fig.canvas.manager.set_window_title(f"{trial_name} — Time Series")

    for i, ch in enumerate(ch_names):
        ax.plot(times_ms, data[i] + offsets[i], linewidth=0.6, label=ch)

    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channel")
    ax.set_title(f"{trial_name} — Amplitude (uV)")
    ax.set_xlim(0, min(window_ms, total_ms))
    ax.grid(True, axis="x", alpha=0.3)

    # Slider for scrolling
    slider_ax = fig.add_axes([0.15, 0.03, 0.7, 0.03])
    max_start = max(0, total_ms - window_ms)
    slider = Slider(slider_ax, "Time (ms)", 0, max_start,
                    valinit=0, valstep=100)

    def update(val):
        start = slider.val
        ax.set_xlim(start, start + window_ms)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    print("Showing interactive time-series plot. Close the window to continue.")
    plt.show()

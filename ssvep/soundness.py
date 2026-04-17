"""Mode 2: Soundness check — offline validation of SSVEP recordings."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from . import config
from .processing import SSVEPClassifier, apply_filters
from .xdf_utils import xdf_to_mne, load_realtime_log


# ---------------------------------------------------------------------------
# Epoch extraction
# ---------------------------------------------------------------------------

def parse_marker_freq(label):
    """Try to extract a target frequency from a marker label string."""
    try:
        f = float(label)
        if f in config.TARGET_FREQS:
            return f
    except (ValueError, TypeError):
        pass
    return None


def extract_epochs(raw, markers, window_sec=config.WINDOW_SEC):
    """Extract EEG epochs from marker-delimited intervals.

    Each epoch starts ONSET_SKIP_SEC after a frequency marker and runs for window_sec.

    Returns:
        list of dicts: {freq, data (n_channels, n_samples), t_start}
    """
    sfreq = raw.info["sfreq"]
    eeg_start = raw.first_samp / sfreq
    total_dur = raw.n_times / sfreq
    skip = config.ONSET_SKIP_SEC

    epochs = []
    for ts, label in markers:
        freq = parse_marker_freq(label)
        if freq is None:
            continue

        t_start = (ts - eeg_start) + skip
        t_end = t_start + window_sec

        if t_start < 0 or t_end > total_dur:
            continue

        s0 = int(t_start * sfreq)
        s1 = int(t_end * sfreq)
        data, _ = raw[:, s0:s1]
        epochs.append({"freq": freq, "data": data, "t_start": t_start})

    return epochs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_accuracy_analysis(epochs, classifier):
    """Run CCA on each epoch and compare to ground truth.

    Returns dict with accuracy, confusion_matrix, per_freq_results.
    """
    results = []
    for ep in epochs:
        detected, corr = classifier.classify(ep["data"])
        results.append({
            "expected": ep["freq"],
            "detected": detected,
            "correlation": corr,
        })

    correct = sum(1 for r in results if r["detected"] == r["expected"])
    accuracy = correct / len(results) if results else 0.0

    # Confusion matrix
    freqs = config.TARGET_FREQS
    n = len(freqs)
    cm = np.zeros((n, n), dtype=int)
    freq_idx = {f: i for i, f in enumerate(freqs)}

    for r in results:
        exp_i = freq_idx.get(r["expected"])
        det_i = freq_idx.get(r["detected"])
        if exp_i is not None and det_i is not None:
            cm[exp_i, det_i] += 1

    return {"accuracy": accuracy, "confusion_matrix": cm, "per_freq_results": results}


def compute_snr(raw, target_freqs=config.TARGET_FREQS, nperseg_sec=2.0):
    """Compute SNR at each target frequency using Welch PSD.

    SNR = power at target freq / mean power in neighboring bands (+-2 Hz, excluding +-0.5 Hz).
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    # Average PSD across channels
    freqs, psd = welch(data, fs=sfreq, nperseg=int(nperseg_sec * sfreq), axis=1)
    psd_mean = psd.mean(axis=0)  # (n_freqs,)

    freq_res = freqs[1] - freqs[0]
    snr = {}
    for tf in target_freqs:
        # Signal: bins within +-0.5 Hz of target
        sig_mask = np.abs(freqs - tf) <= 0.5
        # Noise: bins within +-2 Hz but outside +-0.5 Hz
        noise_mask = (np.abs(freqs - tf) <= 2.0) & (~sig_mask)

        if sig_mask.any() and noise_mask.any():
            sig_power = psd_mean[sig_mask].mean()
            noise_power = psd_mean[noise_mask].mean()
            snr[tf] = 10 * np.log10(sig_power / noise_power) if noise_power > 0 else float("inf")
        else:
            snr[tf] = float("nan")

    return snr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_psd_per_condition(raw, epochs, target_freqs, output_dir):
    """PSD subplot per target frequency, computed from that freq's epochs only."""
    sfreq = raw.info["sfreq"]
    fig, axes = plt.subplots(1, len(target_freqs), figsize=(5 * len(target_freqs), 4), sharey=True)
    if len(target_freqs) == 1:
        axes = [axes]

    for ax, tf in zip(axes, target_freqs):
        # Gather data for this freq
        segments = [ep["data"] for ep in epochs if ep["freq"] == tf]
        if not segments:
            ax.set_title(f"{tf} Hz (no data)")
            continue

        combined = np.concatenate(segments, axis=1)
        freqs, psd = welch(combined, fs=sfreq, nperseg=min(int(2 * sfreq), combined.shape[1]), axis=1)
        psd_mean = psd.mean(axis=0)

        ax.semilogy(freqs, psd_mean)
        # Mark target freq + harmonics
        for h in range(1, config.N_HARMONICS + 1):
            ax.axvline(tf * h, color="red", linestyle="--", alpha=0.6, label=f"{tf*h} Hz" if h == 1 else "")
        ax.set_title(f"{tf} Hz stimulus")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, config.BANDPASS[1] + 5)

    axes[0].set_ylabel("PSD (V²/Hz)")
    fig.suptitle("PSD by Stimulus Condition")
    fig.tight_layout()
    path = os.path.join(output_dir, "psd_per_condition.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_confusion_matrix(cm, labels, output_dir):
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([f"{l} Hz" for l in labels])
    ax.set_yticklabels([f"{l} Hz" for l in labels])
    ax.set_xlabel("Detected")
    ax.set_ylabel("Expected")
    ax.set_title("Confusion Matrix")

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_correlation_distributions(results, output_dir):
    """Box plot of CCA correlations grouped by expected frequency."""
    freqs = config.TARGET_FREQS
    groups = {f: [] for f in freqs}
    for r in results:
        if r["expected"] in groups:
            groups[r["expected"]].append(r["correlation"])

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [groups[f] for f in freqs]
    labels = [f"{f} Hz" for f in freqs]
    ax.boxplot(data, labels=labels)
    ax.axhline(config.CCA_THRESHOLD, color="red", linestyle="--", alpha=0.5, label=f"Threshold ({config.CCA_THRESHOLD})")
    ax.set_ylabel("CCA Correlation")
    ax.set_title("CCA Correlation by Target Frequency")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "correlation_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_detection_histogram(log_data, output_dir):
    """Bar chart of detection frequency counts (for real-time log sub-mode)."""
    counts = {}
    for row in log_data:
        f = row["detected_freq"]
        key = f"{f:.0f} Hz" if f is not None else "None"
        counts[key] = counts.get(key, 0) + 1

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.keys(), counts.values())
    ax.set_ylabel("Count")
    ax.set_title("Detection Frequency Distribution")
    fig.tight_layout()
    path = os.path.join(output_dir, "detection_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_full_psd(raw, target_freqs, output_dir):
    """Single PSD plot of the full recording."""
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    freqs, psd = welch(data, fs=sfreq, nperseg=int(2 * sfreq), axis=1)
    psd_mean = psd.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(freqs, psd_mean)
    for tf in target_freqs:
        ax.axvline(tf, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title("Full Recording PSD")
    ax.set_xlim(0, config.BANDPASS[1] + 5)
    fig.tight_layout()
    path = os.path.join(output_dir, "full_psd.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_log_correlations(log_data, output_dir):
    """Correlation distribution from real-time log, grouped by detected freq."""
    groups = {}
    for row in log_data:
        f = row["detected_freq"]
        key = f if f is not None else "None"
        groups.setdefault(key, []).append(row["correlation"])

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = sorted([k for k in groups if k != "None"]) + (["None"] if "None" in groups else [])
    data = [groups[k] for k in labels]
    display_labels = [f"{k:.0f} Hz" if k != "None" else "None" for k in labels]
    ax.boxplot(data, labels=display_labels)
    ax.axhline(config.CCA_THRESHOLD, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("CCA Correlation")
    ax.set_title("Correlation Distribution (from real-time log)")
    fig.tight_layout()
    path = os.path.join(output_dir, "log_correlation_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_marker_mode(raw, markers, output_dir):
    """Sub-mode A: XDF with event markers — full accuracy analysis."""
    print("\n=== Soundness Check: Marker Mode ===\n")

    raw = apply_filters(raw)
    sfreq = raw.info["sfreq"]
    classifier = SSVEPClassifier(sfreq)

    # Extract epochs
    epochs = extract_epochs(raw, markers)
    print(f"Extracted {len(epochs)} epochs")
    per_freq = {}
    for ep in epochs:
        per_freq.setdefault(ep["freq"], 0)
        per_freq[ep["freq"]] += 1
    for f, n in sorted(per_freq.items()):
        print(f"  {f} Hz: {n} epochs")

    if not epochs:
        print("No valid epochs found. Check marker labels match target frequencies.")
        return

    # Accuracy analysis
    analysis = run_accuracy_analysis(epochs, classifier)
    print(f"\nAccuracy: {analysis['accuracy']:.1%} ({sum(1 for r in analysis['per_freq_results'] if r['detected']==r['expected'])}/{len(analysis['per_freq_results'])})")

    # Per-frequency breakdown
    for tf in config.TARGET_FREQS:
        freq_results = [r for r in analysis["per_freq_results"] if r["expected"] == tf]
        if freq_results:
            correct = sum(1 for r in freq_results if r["detected"] == tf)
            avg_corr = np.mean([r["correlation"] for r in freq_results])
            print(f"  {tf} Hz: {correct}/{len(freq_results)} correct, avg corr={avg_corr:.3f}")

    # SNR
    snr = compute_snr(raw)
    print("\nSNR at target frequencies:")
    for f, s in sorted(snr.items()):
        print(f"  {f} Hz: {s:.1f} dB")

    # Plots
    print("\nGenerating plots...")
    plot_psd_per_condition(raw, epochs, config.TARGET_FREQS, output_dir)
    plot_confusion_matrix(analysis["confusion_matrix"], config.TARGET_FREQS, output_dir)
    plot_correlation_distributions(analysis["per_freq_results"], output_dir)


def run_log_mode(raw, log_data, output_dir):
    """Sub-mode B: Real-time log — no ground truth, distribution analysis only."""
    print("\n=== Soundness Check: Log Mode (no ground truth) ===\n")

    raw = apply_filters(raw)

    # Summary stats from log
    total = len(log_data)
    detected = sum(1 for r in log_data if r["detected_freq"] is not None)
    avg_latency = np.mean([r["latency_ms"] for r in log_data])
    avg_corr = np.mean([r["correlation"] for r in log_data])

    print(f"Total classifications: {total}")
    print(f"Detected (above threshold): {detected}/{total} ({detected/total:.1%})")
    print(f"Avg latency: {avg_latency:.1f} ms")
    print(f"Avg correlation: {avg_corr:.3f}")

    # Frequency counts
    counts = {}
    for r in log_data:
        f = r["detected_freq"]
        counts[f] = counts.get(f, 0) + 1
    print("\nDetection counts:")
    for f in sorted(k for k in counts if k is not None):
        print(f"  {f:.0f} Hz: {counts[f]}")
    if None in counts:
        print(f"  None: {counts[None]}")

    # Plots
    print("\nGenerating plots...")
    plot_full_psd(raw, config.TARGET_FREQS, output_dir)
    plot_detection_histogram(log_data, output_dir)
    plot_log_correlations(log_data, output_dir)


def main():
    parser = argparse.ArgumentParser(description="SSVEP soundness check")
    parser.add_argument("filepath", help="Path to XDF file")
    parser.add_argument("--log", default=None, help="Path to real-time CSV log (activates log mode)")
    parser.add_argument("--output-dir", default="./results", help="Directory for output plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load EEG data
    print(f"Loading {args.filepath}...")
    raw, markers = xdf_to_mne(args.filepath)
    print(f"  {raw.info['nchan']} channels, {raw.n_times} samples @ {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.n_times / raw.info['sfreq']:.1f}s")

    if args.log:
        # Sub-mode B: real-time log
        log_data = load_realtime_log(args.log)
        run_log_mode(raw, log_data, args.output_dir)
    elif markers:
        # Sub-mode A: marker-based
        print(f"  Found {len(markers)} markers")
        run_marker_mode(raw, markers, args.output_dir)
    else:
        print("No markers in XDF and no --log provided.")
        print("Running PSD-only analysis...")
        raw = apply_filters(raw)
        plot_full_psd(raw, config.TARGET_FREQS, args.output_dir)
        snr = compute_snr(raw)
        print("\nSNR at target frequencies:")
        for f, s in sorted(snr.items()):
            print(f"  {f} Hz: {s:.1f} dB")


if __name__ == "__main__":
    main()

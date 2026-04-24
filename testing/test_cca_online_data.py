"""Test CCA classifier accuracy against real SSVEP data from MAT dataset.

Dataset: data_s19_64.mat (subject 19, 64 channels, 1kHz)
H5PY axes: [block, frequency, time_point, channel, condition]
- condition 1 = High-Depth
- occipital channels 56:64
- freq_idx = int(hz) - 1
"""

import os
import sys
import numpy as np
import h5py
from scipy.signal import welch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ssvep.processing import SSVEPClassifier, design_filters, filter_window

MAT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../rover_pipeline/data_s19_64.mat'))
SFREQ = 1000.0
TARGET_FREQS = [7.0, 9.0, 13.0, 17.0]
OCCIPITAL_CHS = slice(56, 64)  # 8 occipital channels
CONDITION = 1  # High-Depth
N_BLOCKS = 6  # blocks to test per frequency


def load_mat():
    if not os.path.exists(MAT_PATH):
        print(f"ERROR: MAT file not found at {MAT_PATH}")
        sys.exit(1)
    return h5py.File(MAT_PATH, 'r')


def extract_window(data_all, freq_hz, block, start_sec, duration_sec):
    """Extract an EEG window for a given frequency, block, and time range.

    Returns: (n_channels, n_samples) numpy array
    """
    freq_idx = int(freq_hz) - 1
    start = int(start_sec * SFREQ)
    end = start + int(duration_sec * SFREQ)
    # axes: [block, frequency, time_point, channel, condition]
    window = data_all[block, freq_idx, start:end, OCCIPITAL_CHS, CONDITION].T
    return np.array(window, dtype=np.float64)


def test_per_frequency_accuracy(window_sec=2.0):
    """Classify multiple windows per frequency across blocks. Report accuracy."""
    print(f"\n{'='*60}")
    print(f"TEST: Per-Frequency Accuracy (window={window_sec}s)")
    print(f"{'='*60}")

    sos_bp, sos_notch = design_filters(SFREQ)
    classifier = SSVEPClassifier(SFREQ, target_freqs=TARGET_FREQS)

    f = load_mat()
    data_all = f['datas']

    results = {freq: {'correct': 0, 'total': 0, 'corrs': []} for freq in TARGET_FREQS}
    confusion = np.zeros((len(TARGET_FREQS), len(TARGET_FREQS)), dtype=int)

    for freq in TARGET_FREQS:
        for block in range(N_BLOCKS):
            # Skip first 0.5s transient, test windows starting at 0.5s and 2.5s
            for start_sec in [0.5, 2.5]:
                try:
                    window = extract_window(data_all, freq, block, start_sec, window_sec)
                    filtered = filter_window(window, sos_bp, sos_notch, causal=False)
                    detected, corr = classifier.classify(filtered)

                    results[freq]['total'] += 1
                    results[freq]['corrs'].append(corr)

                    if detected == freq:
                        results[freq]['correct'] += 1

                    if detected is not None and detected in TARGET_FREQS:
                        true_idx = TARGET_FREQS.index(freq)
                        pred_idx = TARGET_FREQS.index(detected)
                        confusion[true_idx, pred_idx] += 1
                except Exception as e:
                    print(f"  Skip freq={freq} block={block} start={start_sec}: {e}")

    f.close()

    # Print results
    print(f"\n{'Frequency':>10} {'Correct':>8} {'Total':>6} {'Accuracy':>9} {'Mean Corr':>10}")
    print("-" * 50)
    all_pass = True
    for freq in TARGET_FREQS:
        r = results[freq]
        acc = r['correct'] / r['total'] if r['total'] > 0 else 0
        mean_corr = np.mean(r['corrs']) if r['corrs'] else 0
        status = "PASS" if acc >= 0.8 else "FAIL"
        if acc < 0.8:
            all_pass = False
        print(f"{freq:>8.0f} Hz {r['correct']:>8} {r['total']:>6} {acc:>8.1%} {mean_corr:>10.3f}  [{status}]")

    # Confusion matrix
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    header = "         " + "".join(f"{f:>8.0f}" for f in TARGET_FREQS)
    print(header)
    for i, freq in enumerate(TARGET_FREQS):
        row = f"{freq:>7.0f}Hz " + "".join(f"{confusion[i,j]:>8}" for j in range(len(TARGET_FREQS)))
        print(row)

    return all_pass


def test_window_size_sensitivity():
    """Test accuracy with different window sizes. Longer windows should do better."""
    print(f"\n{'='*60}")
    print("TEST: Window Size Sensitivity")
    print(f"{'='*60}")

    f = load_mat()
    data_all = f['datas']

    window_sizes = [1.0, 2.0, 3.0]
    accuracies = {}

    for win_sec in window_sizes:
        sos_bp, sos_notch = design_filters(SFREQ)
        classifier = SSVEPClassifier(SFREQ, target_freqs=TARGET_FREQS)
        correct = 0
        total = 0

        for freq in TARGET_FREQS:
            for block in range(3):  # fewer blocks for speed
                try:
                    window = extract_window(data_all, freq, block, 0.5, win_sec)
                    filtered = filter_window(window, sos_bp, sos_notch, causal=False)
                    detected, corr = classifier.classify(filtered)
                    total += 1
                    if detected == freq:
                        correct += 1
                except Exception:
                    pass

        acc = correct / total if total > 0 else 0
        accuracies[win_sec] = acc
        print(f"  Window {win_sec:.0f}s: {acc:.1%} ({correct}/{total})")

    f.close()

    # Longer windows should generally be >= shorter windows
    if accuracies[3.0] >= accuracies[1.0]:
        print("  [PASS] 3s window >= 1s window accuracy")
    else:
        print("  [WARN] 3s window < 1s window — unexpected")

    return accuracies


def test_snr():
    """Check SNR at target frequencies using Welch PSD."""
    print(f"\n{'='*60}")
    print("TEST: SNR at Target Frequencies")
    print(f"{'='*60}")

    f = load_mat()
    data_all = f['datas']
    sos_bp, sos_notch = design_filters(SFREQ)

    for freq in TARGET_FREQS:
        # Use 3s window from block 0
        window = extract_window(data_all, freq, 0, 0.5, 3.0)
        filtered = filter_window(window, sos_bp, sos_notch, causal=False)

        # Average PSD across channels
        psds = []
        for ch in range(filtered.shape[0]):
            freqs_psd, psd = welch(filtered[ch], fs=SFREQ, nperseg=int(2 * SFREQ))
            psds.append(psd)
        avg_psd = np.mean(psds, axis=0)

        # Find peak near target frequency (±1 Hz)
        mask = (freqs_psd >= freq - 1) & (freqs_psd <= freq + 1)
        peak_power = np.max(avg_psd[mask])
        peak_freq = freqs_psd[mask][np.argmax(avg_psd[mask])]

        # Noise: 2-4 Hz away from target
        noise_mask = ((freqs_psd >= freq - 4) & (freqs_psd <= freq - 2)) | \
                     ((freqs_psd >= freq + 2) & (freqs_psd <= freq + 4))
        noise_power = np.mean(avg_psd[noise_mask])

        snr_db = 10 * np.log10(peak_power / noise_power) if noise_power > 0 else float('inf')
        status = "PASS" if snr_db > 3 else "FAIL"
        print(f"  {freq:>5.0f} Hz: peak at {peak_freq:.1f} Hz, SNR = {snr_db:.1f} dB  [{status}]")

    f.close()


if __name__ == "__main__":
    print("SSVEP CCA Test Suite — Online Dataset (Subject 19)")
    print(f"MAT file: {MAT_PATH}")
    print(f"Target frequencies: {TARGET_FREQS}")
    print(f"Channels: occipital 56-63, Condition: High-Depth")

    passed = test_per_frequency_accuracy()
    test_window_size_sensitivity()
    test_snr()

    print(f"\n{'='*60}")
    if passed:
        print("OVERALL: All per-frequency accuracy tests PASSED (>=80%)")
    else:
        print("OVERALL: Some accuracy tests FAILED (<80%)")
    print(f"{'='*60}")

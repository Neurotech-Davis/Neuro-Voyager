"""Signal processing and CCA classification for SSVEP."""

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt
from sklearn.cross_decomposition import CCA

from . import config


def design_filters(sfreq, bandpass=config.BANDPASS, notch=config.NOTCH):
    """Pre-compute SOS filter coefficients for bandpass + notch."""
    sos_bp = butter(4, bandpass, btype="band", fs=sfreq, output="sos")
    q = 30.0
    b, a = iirnotch(notch, q, fs=sfreq)
    # Convert notch to SOS for consistent interface
    from scipy.signal import tf2sos
    sos_notch = tf2sos(b, a)
    return sos_bp, sos_notch


def filter_window(data, sos_bp, sos_notch, causal=True):
    """Filter a numpy array (n_channels, n_samples) with pre-computed coefficients.

    Args:
        data: EEG data (n_channels, n_samples).
        sos_bp: Bandpass SOS coefficients.
        sos_notch: Notch SOS coefficients.
        causal: If True, use causal filtering (sosfilt). Otherwise sosfiltfilt.

    Returns:
        Filtered data, same shape.
    """
    filt = sosfilt if causal else sosfiltfilt
    out = filt(sos_bp, data, axis=1)
    out = filt(sos_notch, out, axis=1)
    return out


def apply_filters(raw_mne, bandpass=config.BANDPASS, notch=config.NOTCH):
    """Apply bandpass + notch filters to an MNE Raw object (offline mode)."""
    raw_mne.load_data()
    raw_mne.filter(l_freq=bandpass[0], h_freq=bandpass[1], fir_design="firwin", verbose=False)
    raw_mne.notch_filter(freqs=notch, verbose=False)
    return raw_mne


def generate_reference_signals(freq, sfreq, n_samples, n_harmonics=config.N_HARMONICS):
    """Generate sine/cosine reference signals for CCA.

    Returns:
        np.ndarray: (n_harmonics * 2, n_samples)
    """
    t = np.arange(n_samples) / sfreq
    refs = []
    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(refs)


class SSVEPClassifier:
    """CCA-based SSVEP frequency classifier."""

    def __init__(self, sfreq, target_freqs=None, n_harmonics=config.N_HARMONICS,
                 threshold=config.CCA_THRESHOLD):
        self.sfreq = sfreq
        self.target_freqs = target_freqs or config.TARGET_FREQS
        self.n_harmonics = n_harmonics
        self.threshold = threshold
        self.cca = CCA(n_components=1)

    def classify(self, eeg_data):
        """Classify a window of EEG data.

        Args:
            eeg_data: (n_channels, n_samples)

        Returns:
            (winning_freq or None, max_correlation)
            Returns None as freq when below threshold.
        """
        n_samples = eeg_data.shape[1]
        X = eeg_data.T  # (n_samples, n_channels) for CCA

        best_corr = -1.0
        winner = None

        for freq in self.target_freqs:
            Y_refs = generate_reference_signals(freq, self.sfreq, n_samples, self.n_harmonics).T
            self.cca.fit(X, Y_refs)
            X_c, Y_c = self.cca.transform(X, Y_refs)
            corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]

            if corr > best_corr:
                best_corr = corr
                winner = freq

        if best_corr < self.threshold:
            winner = None

        return winner, best_corr

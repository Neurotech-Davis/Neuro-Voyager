"""SSVEP Processor: Cleaning EEG data and classification using CCA."""

import numpy as np
from sklearn.cross_decomposition import CCA
from config import BANDPASS, NOTCH, NHARMONICS, TARGETS

def generate_reference_signals(freq, sfreq, n_samples, n_harmonics=NHARMONICS):
    """
    Generate sine/cosine reference signals for a target frequency and its harmonics.
    
    Args:
        freq (float): Target frequency in Hz.
        sfreq (float): Sampling frequency in Hz.
        n_samples (int): Number of samples in the window.
        n_harmonics (int): Number of harmonics to include.
        
    Returns:
        np.ndarray: Reference signal matrix (n_harmonics * 2, n_samples).
    """
    t = np.arange(n_samples) / sfreq
    refs = []
    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(refs)

class SSVEPClassifier:
    """CCA-based SSVEP classification."""
    
    def __init__(self, sfreq, target_freqs=list(TARGETS.keys()), n_harmonics=NHARMONICS):
        self.sfreq = sfreq
        # Filter out 0.0 (Idle) from targets to generate reference signals
        self.target_freqs = [f for f in target_freqs if f > 0]
        self.n_harmonics = n_harmonics
        self.cca = CCA(n_components=1)
        
    def classify(self, eeg_data):
        """
        Run CCA on a window of EEG data against all target reference signals.
        
        Args:
            eeg_data (np.ndarray): EEG data (n_channels, n_samples).
            
        Returns:
            float: Winning frequency.
            float: Maximum correlation value.
        """
        n_samples = eeg_data.shape[1]
        best_corr = -1.0
        winner = 0.0 # Default to Stop/Idle
        
        # EEG data needs to be (n_samples, n_channels) for CCA
        X = eeg_data.T
        
        for freq in self.target_freqs:
            # Generate references for this window size
            Y_refs = generate_reference_signals(freq, self.sfreq, n_samples, self.n_harmonics).T
            
            # Fit CCA and find correlation
            self.cca.fit(X, Y_refs)
            # Transform to get the canonical variables
            X_c, Y_c = self.cca.transform(X, Y_refs)
            
            # Correlation between the first canonical variables
            corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
            
            if corr > best_corr:
                best_corr = corr
                winner = freq
                
        # Simple thresholding can be added here if needed
        if best_corr < 0.3: # Increased threshold for "no signal"
            winner = 0.0
            
        return winner, best_corr

def apply_filters(raw_mne):
    """
    Applies Bandpass (5-35Hz) and Notch (60Hz) filters to MNE Raw object.
    
    Args:
        raw_mne (mne.io.Raw): Raw EEG data.
        
    Returns:
        mne.io.Raw: Filtered EEG data.
    """
    # Load if not already
    raw_mne.load_data()
    # Phase 1: Signal Pre-Processing
    print(f"Applying filters: Bandpass {BANDPASS}, Notch {NOTCH}")
    raw_mne.filter(l_freq=BANDPASS[0], h_freq=BANDPASS[1], fir_design="firwin", verbose=False)
    raw_mne.notch_filter(freqs=NOTCH, verbose=False)
    return raw_mne

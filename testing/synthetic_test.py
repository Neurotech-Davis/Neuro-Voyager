import numpy as np
import os
import sys

# Add rover_pipeline to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rover_pipeline')))

from ssvep_processor import SSVEPClassifier
from comm_protocol import MotorController
from config import TARGETS

def run_synthetic_test():
    sfreq = 250.0
    duration = 2.0 # seconds
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq
    
    classifier = SSVEPClassifier(sfreq)
    controller = MotorController()
    
    print(f"{'Signal':>10} | {'Winner':>10} | {'Corr':>10} | {'Cmd'}")
    print("-" * 50)
    
    # Test each target
    for target_freq in sorted(TARGETS.keys(), reverse=True):
        # Generate synthetic 8-channel EEG signal
        # Add some noise and harmonics
        if target_freq > 0:
            # Signal: Sine + Harmonic + Noise
            signal = np.sin(2 * np.pi * target_freq * t) + \
                     0.5 * np.sin(2 * np.pi * 2 * target_freq * t) + \
                     0.2 * np.random.randn(n_samples)
        else:
            # Just noise
            signal = 0.5 * np.random.randn(n_samples)
            
        # Broadcast to 8 channels with different weights
        eeg_data = np.tile(signal, (8, 1))
        eeg_data *= np.random.uniform(0.5, 1.5, (8, 1))
        
        winner, corr = classifier.classify(eeg_data)
        cmd = TARGETS.get(winner, 'S')
        
        print(f"{target_freq:>8.1f} Hz | {winner:>8.1f} Hz | {corr:>8.3f} | {cmd}")

if __name__ == "__main__":
    run_synthetic_test()

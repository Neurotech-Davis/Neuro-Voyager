"""Simulate a 'Live' BCI session using the MAT dataset."""

import h5py
import numpy as np
import time
import os
import sys

# Add rover_pipeline to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rover_pipeline')))

from ssvep_processor import SSVEPClassifier
from comm_protocol import MotorController
from config import TARGETS

def simulate_mat_live():
    # File is in rover_pipeline/
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../rover_pipeline/data_s19_64.mat'))
    sfreq = 1000.0
    classifier = SSVEPClassifier(sfreq)
    controller = MotorController() 
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    with h5py.File(file_path, 'r') as f:
        data_all = f['datas']
        
        # Test sequence: 7Hz -> 9Hz -> 11Hz -> 13Hz
        test_sequence = [7.0, 9.0, 11.0, 13.0]
        
        print("\n--- Starting MAT Dataset Live Simulation ---")
        
        for hz in test_sequence:
            print(f"\nStimulating subject with {hz} Hz...")
            freq_idx = int(hz) - 1
            
            # Simulate 3 segments of "looking at the light"
            for second in range(3):
                start_samp = int(second * 1000)
                end_samp = start_samp + 1000 # 1s window
                
                # High-Depth condition (1), occipital channels 56:64
                # H5PY axes: [block, frequency, time_point, channel, condition]
                eeg_window = data_all[0, freq_idx, start_samp:end_samp, 56:64, 1].T
                
                winner, corr = classifier.classify(eeg_window)
                controller.send_command(winner)
                
                print(f"  Time {second}.0s | Detected: {winner:>4} Hz (Corr: {corr:.3f})")
                time.sleep(0.05) # Fast simulation

        print("\nSimulation complete.")

if __name__ == "__main__":
    simulate_mat_live()

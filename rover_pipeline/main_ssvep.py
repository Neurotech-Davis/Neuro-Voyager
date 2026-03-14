"""Main SSVEP Execution script: Simulation of live BCI pipeline."""

import pyxdf
import mne
import numpy as np
import time
import sys
from config import WINDOW_SEC, STEP_SEC, TARGETS, SERIAL_PORT
from ssvep_processor import SSVEPClassifier, apply_filters
from comm_protocol import MotorController

def load_xdf_to_mne(file_path):
    """Loads EEG stream from XDF and converts to MNE Raw object."""
    streams, header = pyxdf.load_xdf(file_path)
    
    # Find EEG stream
    eeg_stream = None
    for s in streams:
        if s['info']['type'][0] == 'EEG':
            eeg_stream = s
            break
            
    if eeg_stream is None:
        raise ValueError("No EEG stream found in XDF file.")
        
    data = eeg_stream['time_series'].T # MNE expects (n_channels, n_samples)
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    
    # Scale if necessary (OpenBCI data is usually in uV, MNE expects Volts)
    # Most XDF OpenBCI streams are already scaled or in uV. 
    # Let's assume it needs scaling if values are large.
    if np.max(np.abs(data)) > 1.0:
        data = data * 1e-6 # uV to Volts
        
    n_channels = data.shape[0]
    ch_names = [f"Ch{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    return raw

def run_pipeline(file_path, port=SERIAL_PORT):
    # 1. Load data
    print(f"Loading data from {file_path}...")
    raw = load_xdf_to_mne(file_path)
    sfreq = raw.info['sfreq']
    
    # 2. Phase 1: Pre-processing (Filters)
    raw_filtered = apply_filters(raw)
    
    # 3. Setup Classifier and Motor Controller
    classifier = SSVEPClassifier(sfreq)
    controller = MotorController(port=port)
    controller.connect()
    
    # 4. Phase 2 & 3: Sliding Window Simulation
    print("Starting SSVEP Simulation...")
    
    # Window and step in samples
    window_samples = int(WINDOW_SEC * sfreq)
    step_samples = int(STEP_SEC * sfreq)
    total_samples = raw_filtered.n_times
    
    try:
        for start in range(0, total_samples - window_samples, step_samples):
            end = start + window_samples
            # Extract window
            window_data, times = raw_filtered[:, start:end]
            
            # Phase 2: Classification (Math)
            winner, corr = classifier.classify(window_data)
            
            # Phase 3: Communication Protocol
            # Map frequency to command and send (with state management)
            controller.send_command(winner)
            
            print(f"Window {start/sfreq:5.1f}s - {end/sfreq:5.1f}s | Winner: {winner:>4} Hz (corr: {corr:.3f})")
            
            # Simulate real-time processing time
            time.sleep(0.1) 
            
    except KeyboardInterrupt:
        print("\nPipeline stopped.")
    finally:
        controller.disconnect()

if __name__ == "__main__":
    file_to_use = "data/Trial1.xdf"
    port_to_use = SERIAL_PORT
    
    if len(sys.argv) > 1:
        port_to_use = sys.argv[1]
    
    run_pipeline(file_to_use, port_to_use)

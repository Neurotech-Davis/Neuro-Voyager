def filter_eeg_data(raw_eeg_data):
    raw_eeg_data.load_data()
    raw_eeg_data.filter(l_freq=5.0, h_freq=35.0, fir_design="firwin")
    raw_eeg_data.notch_filter(freqs=60.0)
    return raw_eeg_data

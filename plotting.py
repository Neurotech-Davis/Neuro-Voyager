from turtle import color

import matplotlib.pyplot as plt

def plot_eeg_data(eeg_data_object, plot_title, window_milliseconds=5000.0):
    # Plot raw data
    eeg_data_object.plot(
        duration=10,               # initial time window (seconds)
        n_channels=5,              # how many channels to show initially
        scalings='auto',           # auto vertical scaling per channel
        title=f"{plot_title} - Time Series",
        show_scrollbars=True,      # try to show them
        show_scalebars=True,       # shows microV / unit scale bars
        block=False                # Don't block yet, so we can show PSD too
    )

    color = {ch: plt.cm.tab10(i % 10) for i, ch in enumerate(eeg_data_object.ch_names)}

    eeg_data_object.plot(color=color)

    # Step 7: Plot the Power Spectral Density (PSD)
    print("Computing and plotting Power Spectral Density (PSD)...")
    psd_figure = eeg_data_object.compute_psd(fmax=30).plot(show=False)
    psd_figure.canvas.manager.set_window_title(f"{plot_title} - PSD")

    # Add vertical lines at target frequencies
    target_freqs = [7, 9, 13, 17]
    for ax in psd_figure.axes:
        for freq in target_freqs:
            ax.axvline(x=freq, color='r', linestyle='--', alpha=0.5)

    # Block execution here to keep all windows open
    plt.show(block=True)
    # figure.savefig('raw_plot.png')
    # Do the same for filtered.

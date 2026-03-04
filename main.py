"""CLI entrypoint for the SSVEP pipeline.

Usage:
    python main.py data/Trial2.xdf                                 # full pipeline
    python main.py output/Trial2_cleaned.fif --skip-preprocessing   # analysis only
"""

import argparse
import os

from analyze import compute_psd_per_target, plot_results, plot_time_series, segment_epochs
from preprocess import clean, load_raw


def main():
    parser = argparse.ArgumentParser(description="SSVEP preprocessing + analysis pipeline")
    parser.add_argument("filepath", help="Path to .xdf or .fif file")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip filtering/ICA (use with already-cleaned .fif)")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    trial_name = os.path.splitext(os.path.basename(args.filepath))[0]

    print(f"Loading {args.filepath}...")
    raw = load_raw(args.filepath)

    if not args.skip_preprocessing:
        print("Cleaning (bandpass, notch, CAR, ICA)...")
        raw = clean(raw)
        os.makedirs(args.output_dir, exist_ok=True)
        fif_path = os.path.join(args.output_dir, f"{trial_name}_cleaned.fif")
        raw.save(fif_path, overwrite=True)
        print(f"Saved: {fif_path}")

    plot_time_series(raw, trial_name)

    print("Segmenting epochs from markers...")
    epochs = segment_epochs(raw)
    print(f"  Found {len(epochs)} epochs")

    if not epochs:
        print("ERROR: No epochs to analyze. Add stimulus markers to your data.")
        print("  Expected: LSL markers like 'stim_7', 'stim_15', etc.")
        return

    print("Computing PSD per target frequency...")
    freq_data = compute_psd_per_target(
        raw.get_data(), epochs, raw.info["sfreq"], raw.ch_names
    )

    print("Plotting results...")
    plot_results(freq_data, raw.ch_names, trial_name, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()

"""Mode 1: Real-time SSVEP classification from live LSL stream."""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import pylsl

from . import config
from .processing import SSVEPClassifier, design_filters, filter_window


class RealtimeClassifier:
    """Connects to LSL EEG stream, runs sliding-window CCA, logs results."""

    def __init__(self, stream_type=config.EEG_STREAM_TYPE, output_dir="."):
        # Resolve EEG stream
        print(f"Resolving LSL stream (type={stream_type})...")
        streams = pylsl.resolve_byprop("type", stream_type, timeout=10.0)
        if not streams:
            raise RuntimeError(f"No LSL stream of type '{stream_type}' found.")

        self.inlet = pylsl.StreamInlet(streams[0], max_buflen=int(config.WINDOW_SEC * 2))
        info = self.inlet.info()
        self.sfreq = info.nominal_srate()
        self.n_channels = info.channel_count()

        print(f"Connected: {info.name()} | {self.n_channels} ch @ {self.sfreq} Hz")

        # Buffer sized for one analysis window
        self.window_samples = int(config.WINDOW_SEC * self.sfreq)
        self.step_samples = int(config.STEP_SEC * self.sfreq)
        self.buffer = np.zeros((self.n_channels, self.window_samples))
        self.samples_in_buffer = 0

        # Classifier + filters
        self.classifier = SSVEPClassifier(self.sfreq)
        self.sos_bp, self.sos_notch = design_filters(self.sfreq)

        # CSV output
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"realtime_{ts}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "detected_freq", "correlation", "latency_ms"])

    def run(self):
        """Main acquisition + classification loop. Ctrl+C to stop."""
        print(f"Logging to {self.csv_path}")
        print(f"Window: {config.WINDOW_SEC}s | Step: {config.STEP_SEC}s")
        print("Filling buffer...\n")

        samples_since_classify = 0

        try:
            while True:
                # Pull available samples
                chunk, timestamps = self.inlet.pull_chunk(timeout=0.1)
                if not chunk:
                    continue

                chunk = np.array(chunk).T  # (n_channels, n_new_samples)
                n_new = chunk.shape[1]
                pull_time = pylsl.local_clock()

                # Shift buffer left and append new data
                if n_new >= self.window_samples:
                    self.buffer = chunk[:, -self.window_samples:]
                    self.samples_in_buffer = self.window_samples
                else:
                    self.buffer = np.roll(self.buffer, -n_new, axis=1)
                    self.buffer[:, -n_new:] = chunk
                    self.samples_in_buffer = min(self.samples_in_buffer + n_new, self.window_samples)

                samples_since_classify += n_new

                # Wait until buffer is full
                if self.samples_in_buffer < self.window_samples:
                    continue

                # Classify every step_samples
                if samples_since_classify < self.step_samples:
                    continue
                samples_since_classify = 0

                # Filter + classify
                filtered = filter_window(self.buffer.copy(), self.sos_bp, self.sos_notch, causal=True)
                winner, corr = self.classifier.classify(filtered)
                classify_time = pylsl.local_clock()

                latency_ms = (classify_time - pull_time) * 1000.0

                # Log
                freq_str = f"{winner}" if winner is not None else ""
                self.csv_writer.writerow([
                    f"{timestamps[-1]:.6f}",
                    freq_str,
                    f"{corr:.4f}",
                    f"{latency_ms:.1f}",
                ])
                self.csv_file.flush()

                label = f"{winner:.0f} Hz" if winner else "---"
                print(f"  {label}  (r={corr:.3f}, latency={latency_ms:.1f}ms)")

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.csv_file.close()
            print(f"Log saved: {self.csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Real-time SSVEP classification via LSL")
    parser.add_argument("--output-dir", default=".", help="Directory for CSV output")
    parser.add_argument("--stream-type", default=config.EEG_STREAM_TYPE, help="LSL stream type")
    args = parser.parse_args()

    rt = RealtimeClassifier(stream_type=args.stream_type, output_dir=args.output_dir)
    rt.run()


if __name__ == "__main__":
    main()

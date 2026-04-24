"""SSVEP Controller GUI — WASD keys classify MAT dataset signals via CCA."""

import os
import sys
import tkinter as tk
from tkinter import ttk
import numpy as np
import h5py
from scipy.signal import welch

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ssvep.processing import SSVEPClassifier, design_filters, filter_window
from ssvep.config import TARGET_FREQS, WASD_MAP
from gui.serial_utils import ArduinoController, find_usb_ports

MAT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../rover_pipeline/data_s19_64.mat'))
SFREQ = 1000.0
OCCIPITAL_CHS = slice(56, 64)
CONDITION = 1  # High-Depth
WINDOW_SEC = 2.0

DIRECTION_LABELS = {
    'w': 'Forward',  'a': 'Left',
    's': 'Backward', 'd': 'Right',
}

CMD_NAMES = {'w': 'Forward', 'a': 'Left', 's': 'Backward', 'd': 'Right', 'e': 'Stop'}


class SSVEPGui:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroVoyager SSVEP Controller")
        self.root.geometry("1000x1300")
        self.root.resizable(True, True)
        self.root.minsize(800, 1000)

        self.arduino = ArduinoController()
        self.classifier = SSVEPClassifier(SFREQ, target_freqs=TARGET_FREQS)
        self.sos_bp, self.sos_notch = design_filters(SFREQ)
        self.mat_file = None
        self.data_all = None
        self.block_counters = {}  # track which block per freq

        self.held_key = None          # currently held WASD key
        self.stop_timer = None         # scheduled stop callback id
        self.psd_timer = None          # live PSD update timer
        self.psd_offset = 0            # sliding window offset into MAT data
        self.psd_block = 0             # current block for live PSD

        self._load_mat()
        self._build_ui()
        self.root.bind('<KeyPress>', self._on_keypress)
        self.root.bind('<KeyRelease>', self._on_keyrelease)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_mat(self):
        if not os.path.exists(MAT_PATH):
            print(f"WARNING: MAT file not found at {MAT_PATH}")
            return
        self.mat_file = h5py.File(MAT_PATH, 'r')
        self.data_all = self.mat_file['datas']
        n_blocks = self.data_all.shape[0]
        print(f"Loaded MAT dataset: {self.data_all.shape}, {n_blocks} blocks")
        for freq, _ in WASD_MAP.values():
            self.block_counters[freq] = 0

    def _build_ui(self):
        # --- Serial connection frame ---
        serial_frame = ttk.LabelFrame(self.root, text="Arduino Connection", padding=10)
        serial_frame.pack(fill='x', padx=10, pady=5)

        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(serial_frame, textvariable=self.status_var, font=('Menlo', 14)).pack(anchor='w')

        port_row = ttk.Frame(serial_frame)
        port_row.pack(fill='x', pady=5)

        ttk.Label(port_row, text="Port:").pack(side='left')
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_row, textvariable=self.port_var, width=30)
        self.port_combo.pack(side='left', padx=5)

        ttk.Button(port_row, text="Refresh", command=self._refresh_ports).pack(side='left', padx=2)
        self.connect_btn = ttk.Button(port_row, text="Connect", command=self._toggle_connect)
        self.connect_btn.pack(side='left', padx=2)
        self.test_btn = ttk.Button(port_row, text="Test", command=self._test_connection, state='disabled')
        self.test_btn.pack(side='left', padx=2)

        self.arduino_resp_var = tk.StringVar(value="Arduino: —")
        ttk.Label(serial_frame, textvariable=self.arduino_resp_var, font=('Menlo', 12),
                  foreground='gray').pack(anchor='w')

        self._refresh_ports()

        # --- WASD display ---
        wasd_frame = ttk.LabelFrame(self.root, text="Controls (press WASD)", padding=10)
        wasd_frame.pack(fill='x', padx=10, pady=5)

        wasd_text = (
            "          W (7 Hz - Forward)\n"
            "A (9 Hz - Left)          D (17 Hz - Right)\n"
            "          S (13 Hz - Backward)"
        )
        self.wasd_label = tk.Label(wasd_frame, text=wasd_text, font=('Menlo', 18),
                                   justify='center', bg='#f0f0f0', padx=30, pady=15)
        self.wasd_label.pack(fill='x')

        # --- Classification results ---
        result_frame = ttk.LabelFrame(self.root, text="Classification Result", padding=10)
        result_frame.pack(fill='x', padx=10, pady=5)

        self.key_var = tk.StringVar(value="Key: —")
        self.freq_var = tk.StringVar(value="Detected: —")
        self.corr_var = tk.StringVar(value="Correlation: —")
        self.cmd_var = tk.StringVar(value="Command: —")

        for var in [self.key_var, self.freq_var, self.corr_var, self.cmd_var]:
            ttk.Label(result_frame, textvariable=var, font=('Menlo', 16)).pack(anchor='w', pady=2)

        # --- Live EEG + PSD plots ---
        plot_frame = ttk.LabelFrame(self.root, text="Live EEG Signal", padding=5)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(9, 5), dpi=100)
        self.ax_eeg = self.fig.add_subplot(211)
        self.ax_psd = self.fig.add_subplot(212)

        self.ax_eeg.set_xlabel("Time (s)")
        self.ax_eeg.set_ylabel("Amplitude (uV)")
        self.ax_eeg.set_title("EEG — Press WASD")

        self.ax_psd.set_xlabel("Frequency (Hz)")
        self.ax_psd.set_ylabel("Power (dB)")
        self.ax_psd.set_xlim(0, 40)
        self.ax_psd.set_title("PSD")
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # --- Log ---
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=5)
        log_frame.pack(fill='both', padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_frame, height=8, font=('Menlo', 13), state='disabled')
        self.log_text.pack(fill='both')

    def _refresh_ports(self):
        ports = find_usb_ports()
        self.port_combo['values'] = ports if ports else ['No USB ports found']
        if ports:
            self.port_var.set(ports[0])

    def _toggle_connect(self):
        if self.arduino.connected:
            self.arduino.disconnect()
            self.status_var.set("Disconnected")
            self.connect_btn.config(text="Connect")
            self.test_btn.config(state='disabled')
            self.arduino_resp_var.set("Arduino: —")
            self._log("Disconnected from Arduino")
        else:
            port = self.port_var.get()
            self._log(f"Connecting to {port}...")
            self.root.update()
            ok, msg = self.arduino.connect(port)
            if ok:
                self.status_var.set(f"Connected: {port}")
                self.connect_btn.config(text="Disconnect")
                self.test_btn.config(state='normal')
            else:
                self.status_var.set(f"Error: {msg}")
            self._log(msg)

    def _test_connection(self):
        """Send stop command to Arduino and check for echo-back response."""
        self._log("Testing Arduino connection...")
        self.root.update()
        ok, msg = self.arduino.test_connection()
        self.arduino_resp_var.set(f"Arduino: {msg}")
        if ok:
            self.status_var.set(f"VERIFIED: {self.port_var.get()}")
        self._log(f"Test: {msg}")

    def _on_keypress(self, event):
        key = event.char.lower()
        if key not in WASD_MAP or self.data_all is None:
            return

        # Cancel any pending stop — this is a held key repeat
        if self.stop_timer is not None:
            self.root.after_cancel(self.stop_timer)
            self.stop_timer = None

        # Already holding this key — ignore OS repeat
        if self.held_key == key:
            return

        self.held_key = key
        freq, cmd = WASD_MAP[key]
        direction = DIRECTION_LABELS[key]

        # Extract EEG window — cycle through blocks
        freq_idx = int(freq) - 1
        block = self.block_counters[freq] % self.data_all.shape[0]
        self.block_counters[freq] += 1

        start = int(0.5 * SFREQ)  # skip 0.5s transient
        end = start + int(WINDOW_SEC * SFREQ)

        try:
            window = self.data_all[block, freq_idx, start:end, OCCIPITAL_CHS, CONDITION].T
            window = np.array(window, dtype=np.float64)
        except Exception as e:
            self._log(f"Data extraction error: {e}")
            return

        # Filter + classify
        filtered = filter_window(window, self.sos_bp, self.sos_notch, causal=False)
        detected, corr = self.classifier.classify(filtered)

        # Update display
        self.key_var.set(f"Key: {key.upper()} ({direction}) [HELD]")
        det_str = f"{detected:.0f} Hz" if detected else "None"
        self.freq_var.set(f"Detected: {det_str}")
        self.corr_var.set(f"Correlation: {corr:.3f}")

        match = "MATCH" if detected == freq else "MISMATCH"

        # Send command to Arduino
        sent, send_msg = self.arduino.send(cmd, force=True)
        cmd_name = CMD_NAMES.get(cmd, cmd)
        self.cmd_var.set(f"Command: {cmd} ({cmd_name})")

        self._log(f"[{key.upper()}] {freq}Hz → detected {det_str} (r={corr:.3f}) [{match}] | {send_msg}")

        # Check for Arduino echo-back
        if self.arduino.connected:
            self.root.after(200, self._check_arduino_response)

        # Start live PSD feed — slides through MAT data while key is held
        self.psd_offset = int(0.5 * SFREQ)  # start after transient
        self.psd_block = block
        self._update_live_psd()

    def _update_live_psd(self):
        """Live feed: slide through MAT data while key is held, update EEG + PSD plots."""
        if self.held_key is None or self.data_all is None:
            return

        freq, cmd = WASD_MAP[self.held_key]
        freq_idx = int(freq) - 1
        n_samples = int(WINDOW_SEC * SFREQ)
        step = int(0.25 * SFREQ)  # slide 250ms per update

        start = self.psd_offset
        end = start + n_samples

        # Wrap around to next block if past end of data
        max_samples = self.data_all.shape[2]
        if end > max_samples:
            self.psd_block = (self.psd_block + 1) % self.data_all.shape[0]
            self.psd_offset = int(0.5 * SFREQ)
            start = self.psd_offset
            end = start + n_samples

        try:
            window = self.data_all[self.psd_block, freq_idx, start:end, OCCIPITAL_CHS, CONDITION].T
            window = np.array(window, dtype=np.float64)
        except Exception:
            return

        filtered = filter_window(window, self.sos_bp, self.sos_notch, causal=False)

        # --- EEG time series (channel average) ---
        self.ax_eeg.clear()
        t = np.arange(filtered.shape[1]) / SFREQ
        avg_signal = np.mean(filtered, axis=0)
        self.ax_eeg.plot(t, avg_signal, 'k-', linewidth=0.8)
        self.ax_eeg.set_xlabel("Time (s)")
        self.ax_eeg.set_ylabel("Amplitude")
        time_sec = start / SFREQ
        self.ax_eeg.set_title(f"EEG — {freq:.0f} Hz stimulus | t={time_sec:.1f}s block={self.psd_block}")

        # --- PSD ---
        self.ax_psd.clear()
        psds = []
        for ch in range(filtered.shape[0]):
            freqs_psd, psd = welch(filtered[ch], fs=SFREQ, nperseg=int(SFREQ))
            psds.append(psd)
        avg_psd = np.mean(psds, axis=0)
        psd_db = 10 * np.log10(avg_psd + 1e-20)
        self.ax_psd.plot(freqs_psd, psd_db, 'b-', linewidth=1.2)

        # Mark target frequencies
        colors = {7.0: '#2ecc71', 9.0: '#e67e22', 13.0: '#e74c3c', 17.0: '#9b59b6'}
        for f in TARGET_FREQS:
            color = colors.get(f, 'gray')
            lw = 2.5 if f == freq else 1.0
            ls = '-' if f == freq else '--'
            self.ax_psd.axvline(f, color=color, linestyle=ls, linewidth=lw,
                               alpha=0.9 if f == freq else 0.5,
                               label=f"{f:.0f} Hz")

        self.ax_psd.set_xlabel("Frequency (Hz)")
        self.ax_psd.set_ylabel("Power (dB)")
        self.ax_psd.set_xlim(0, 40)
        self.ax_psd.set_title(f"PSD — {freq:.0f} Hz")
        self.ax_psd.legend(loc='upper right', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

        # Advance window and schedule next update
        self.psd_offset += step
        self.psd_timer = self.root.after(250, self._update_live_psd)

    def _on_keyrelease(self, event):
        """Schedule stop on key release. Delay filters out macOS key-repeat fake releases."""
        key = event.char.lower()
        if key not in WASD_MAP or key != self.held_key:
            return

        # Schedule stop after 50ms — if another KeyPress arrives before then,
        # it's a key-repeat and we cancel this stop
        self.stop_timer = self.root.after(50, self._do_stop, key)

    def _do_stop(self, key):
        """Actually send stop command after confirming key was truly released."""
        self.stop_timer = None
        self.held_key = None
        # Stop live PSD feed
        if self.psd_timer is not None:
            self.root.after_cancel(self.psd_timer)
            self.psd_timer = None
        sent, msg = self.arduino.send('e', force=True)
        self.cmd_var.set("Command: e (Stop)")
        self.key_var.set("Key: —")
        self._log(f"[{key.upper()} released] Stop | {msg}")
        if self.arduino.connected:
            self.root.after(200, self._check_arduino_response)

    def _check_arduino_response(self):
        """Read and display any echo-back from Arduino."""
        resp = self.arduino.read_response(timeout=0.2)
        if resp:
            self.arduino_resp_var.set(f"Arduino: {resp}")
            self._log(f"  ← Arduino: {resp}")

    def _log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def _on_close(self):
        if self.psd_timer is not None:
            self.root.after_cancel(self.psd_timer)
        self.arduino.disconnect()
        if self.mat_file:
            self.mat_file.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    SSVEPGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()

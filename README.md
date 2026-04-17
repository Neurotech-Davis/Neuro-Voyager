# Neuro-Voyager

Projects '25–'26 Beginning Track Team 3: Neuro-Voyager  
BCI-controlled rover using SSVEP (Steady-State Visual Evoked Potentials) classified via CCA.

---

## Architecture

```
NeuroVoyager/
├── ssvep/                     # Main SSVEP pipeline (package)
│   ├── config.py              # Shared constants (freqs, filter params, CCA settings)
│   ├── processing.py          # SSVEPClassifier (CCA), filter design, reference signals
│   ├── xdf_utils.py           # XDF loading, MNE conversion, real-time log parsing
│   ├── realtime.py            # Mode 1: live LSL stream classification
│   ├── soundness.py           # Mode 2: offline validation & analysis
│   └── __main__.py            # Entry point for `python -m ssvep`
│
├── rover_pipeline/            # Legacy rover control pipeline
│   ├── main_ssvep.py          # XDF simulation → serial rover commands
│   ├── ssvep_processor.py     # Classifier (older version)
│   ├── comm_protocol.py       # Serial motor controller
│   └── config.py              # Pipeline config
│
├── main.py                    # Standalone EEG viewer (XDF/FIF → filter → plot)
├── filter.py                  # MNE bandpass + notch filter helper
└── plotting.py                # MNE Raw plot helper
```

### Signal Processing

- **Bandpass**: 5–35 Hz (4th-order Butterworth)
- **Notch**: 60 Hz power-line
- **Classifier**: CCA against sine/cosine reference signals at target frequencies + harmonics
- **Target frequencies**: 7, 9, 13, 17 Hz (configurable in `ssvep/config.py`)
- **Window**: 2s sliding, 0.5s step, CCA threshold 0.3

---

## Usage

### Mode 1 — Real-time Classification (live LSL stream)

Connects to an LSL EEG stream, runs sliding-window CCA, logs results to CSV.

```bash
python -m ssvep.realtime
python -m ssvep.realtime --output-dir ./logs --stream-type EEG
```

Output: `realtime_YYYYMMDD_HHMMSS.csv` with columns `timestamp, detected_freq, correlation, latency_ms`.

### Mode 2 — Soundness Check (offline validation)

#### Sub-mode A: Marker mode (XDF with event markers)

Full accuracy analysis — extracts epochs per stimulus frequency, runs CCA, reports accuracy + SNR.

```bash
python -m ssvep.soundness recording.xdf
python -m ssvep.soundness recording.xdf --output-dir ./results
```

Outputs: `psd_per_condition.png`, `confusion_matrix.png`, `correlation_distributions.png`

#### Sub-mode B: Log mode (real-time CSV log)

Distribution analysis from a prior real-time session — no ground truth required.

```bash
python -m ssvep.soundness recording.xdf --log realtime_20250101_120000.csv
```

Outputs: `full_psd.png`, `detection_histogram.png`, `log_correlation_distributions.png`

#### Sub-mode C: PSD-only (no markers, no log)

Falls back to PSD + SNR report when no markers or log are present.

```bash
python -m ssvep.soundness recording.xdf
```

### Standalone EEG Viewer

Quick offline viewer — loads XDF or FIF, optionally filters, plots with MNE.

```bash
python main.py path/to/recording.xdf
python main.py path/to/recording.xdf --raw   # skip filtering
```

### Legacy Rover Pipeline

Simulates real-time BCI pipeline from a saved XDF, sends commands over serial.

```bash
cd rover_pipeline
python main_ssvep.py [/dev/ttyUSBx]
```

---

## Configuration

All tunable parameters in `ssvep/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `TARGET_FREQS` | `[7, 9, 13, 17]` Hz | SSVEP stimulus frequencies |
| `BANDPASS` | `(5.0, 35.0)` Hz | Filter passband |
| `NOTCH` | `60.0` Hz | Power-line notch |
| `N_HARMONICS` | `3` | CCA reference harmonics |
| `WINDOW_SEC` | `2.0` s | Classification window length |
| `STEP_SEC` | `0.5` s | Sliding window step |
| `CCA_THRESHOLD` | `0.3` | Min correlation to report detection |
| `ONSET_SKIP_SEC` | `0.5` s | Skip after stimulus onset (soundness mode) |

---

## Dependencies

```
mne
pyxdf
pylsl
numpy
scipy
scikit-learn
matplotlib
```

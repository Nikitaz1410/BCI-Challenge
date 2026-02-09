# BCI Challenge: Online EEG Motor Imagery Decoder

> **1st Place Solution** — *Praktikum: Developing Reliable Decoders for a Brain-Computer Interface (WiSe 2025/26)*
>
> Chair of Engineering Resilient Cognitive Systems, TUM School of Computation, Information and Technology

**Team:** Nikita Zubairov, Iustin Curcean, Amal Ben Slimen and Daria Bondareva

---

## Overview

This project implements a complete real-time Brain-Computer Interface (BCI) system for decoding EEG Motor Imagery (MI) signals into three classes — **Left Hand**, **Right Hand**, and **Rest** — and using the decoded commands to play the Chrome Dino Game in real time (Chrome Dino Game implementation not included).

Key challenges addressed:
- **Non-stationarity** of EEG signals across sessions
- **Low signal-to-noise ratio** inherent to non-invasive EEG
- **No calibration** — the decoder works without per-session recalibration
- **Real-time constraint** — reliable control within a 3-second decision window
- **Sparse training data** — only a single subject with limited recording sessions

We explored a broad range of models spanning traditional machine learning, Riemannian geometry, and deep learning, evaluating all candidates via **Leave-One-Session-Out (LOSO) cross-validation**. Our final online submission used four models: **Adaptive Riemannian Classifier**, **Adaptive MIRepNet**, **ShallowConvNet**, and **CSP-SVM**.

## Pipeline Architecture

### Offline Pipeline

```
.xdf Files ──► Butterworth Filter ──► Autoreject [2] ──► Sliding Windows ──► Train Model
                 IIR, Causal (SOS)                       1.0s window
                 8th Order, 8–30 Hz                      0.128s step
                 82.98 ms delay
```

### Online (Real-Time) Pipeline

```
                  ┌──────────────┐
LSL Stream ─────► │  EEG Chunk   │
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │   Stateful   │     Filter states persist
                  │   Filter     │     across chunks
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │  1s Rolling  │     Sliding window buffer
                  │   Buffer     │
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │  Autoreject  │     Artifact detection
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │    Model     │     Classification
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │  Transfer    │     Moving-average smoothing
                  │  Function    │──────► Chrome Dino Game (UDP)
                  └──────────────┘
```

---

## Models

We implemented and compared models across four families, selecting the best performers via **11-fold session-wise LOSO cross-validation** on our target subject's data (11 sessions, 23 recording files).

### Traditional Baselines

```
EEG Signal ──► Standardize ──► Feature Extraction ──► Classifier ──► Class
                                  ├─ CSP                ├─ LDA
                                  └─ Log Band Power     ├─ SVM
                                     (Welch PSD)        ├─ Logistic Regression
                                                        └─ Random Forest
```

Modular pipeline combining Common Spatial Patterns and/or Welch log-bandpower features with multiple classifiers. Based on our results, the **CSP-SVM** variant was selected to be part the final online submission for this model family.

### Adaptive Riemannian Classification [3]

```
EEG Signal ──► Spatial Covariance ──► Recentering ──► Riemann Distance ──► Class
               Matrices                                 (FgMDM)
                                         ▲
                                         │
                                  Online Adaptive
                                  Recentering
```

Classifies EEG trials in the space of symmetric positive-definite (SPD) covariance matrices using the Fisher Geodesic Minimum Distance to Mean (FgMDM) classifier. Session-wise **recentering** projects each session's data to a common manifold origin, reducing inter-session variability. During online use, the recentering centroid is **adapted unsupervised** via geodesic interpolation, making the model robust to distribution shifts without labeled data.

### Deep Learning Models

We benchmarked seven convolutional architectures for MI-EEG decoding:

| Architecture | Reference |
|---|---|
| **ShallowConvNet** | Schirrmeister et al. [7] |
| **DeepConvNet** | Schirrmeister et al. [7] |
| **EEGNet** | Lawhern et al. [10] |
| **FBCNet** | Mane et al. [6] |
| **Conformer** | Song et al. [5] |
| **IFNet** | Wang et al. [9] |
| **ADFCNN** | Tao et al. [4] |

**ShallowConvNet** was selected to be part of our final online submission.

### MIRepNet — Foundation Model [11]

A pre-trained EEG foundation model fine-tuned on our target subject. MIRepNet uses masked-latent-modeling (MLM) pre-training on large-scale EEG datasets. We also developed an **Adaptive MIRepNet** variant that updates the Euclidean Alignment Reference online.

### Additional: Combined Adaptive LDA

An ensemble of a 2-stage hierarchical LDA (Rest vs. Movement, then Left vs. Right) and a standard 3-class LDA with confidence-based model selection and online EMA adaptation of class means. Inspired by Wu et al.'s adaptive LDA approach.

---

## Final Online Submission

The following four models were deployed for the online Dino Game challenge:

| Model |
|---|
| **Adaptive Riemannian** | 
| **MIRepNet** |
| **ShallowConvNet** |
| **CSP-SVM** |

---

## Preprocessing

| Step | Details |
|---|---|
| **Bandpass Filter** | 8–30 Hz, Butterworth IIR, causal (SOS), 8th order, 82.98 ms group delay. Stateful for online continuity |
| **Channel Selection** | 11 of 16 channels retained: F3, Fz, F4, C3, Cz, C4, P3, Pz, P4, PO7, PO8. Removed: Fp1, Fp2, T7, T8, Oz (noise-prone) (except for MIRepNet) |
| **Artifact Removal** | Autoreject [2] for offline calibration. Online: multi-criteria checks (amplitude, variance, gradient, channel consistency) with thresholds calibrated offline |
| **Windowing** | 1.0 s windows (250 samples at 250 Hz) with 0.128 s step (32 samples) |

---

## Cross-Validation

- **Method:** 11-fold session-wise Leave-One-Session-Out (LOSO)
- **Grouping:** Sessions extracted from BIDS filenames (`sub-XXX_ses-YYY`) — multiple runs from the same session are never split across folds
- **Data integrity:** Artifact rejection thresholds are fit on training data only (no leakage)

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall classification accuracy |
| **Balanced Accuracy** | Accounts for class imbalance |
| **Macro F1** | Harmonic mean of precision and recall (macro-averaged) |
| **ECE** | Expected Calibration Error — probability calibration quality (10 bins) |
| **Brier Score** | Multiclass Brier score — penalizes confident wrong predictions |
| **ITR** | Information Transfer Rate (bits/min) via Wolpaw's formula |
| **Latency** | Filter delay, inference time, and total end-to-end latency |

---

## Decision & Game Control

- **Probability smoothing:** Moving-average buffer over consecutive predictions (configurable buffer size)
- **Threshold:** Classification accepted when smoothed confidence exceeds the configured threshold
- **Communication:** UDP to Chrome Dino Game with JSON payload (`{"prediction": "ARROW LEFT", "marker_recent": "ARROW LEFT ONSET"}`)
- **Commands:** `CIRCLE` (rest), `ARROW LEFT` (left hand MI), `ARROW RIGHT` (right hand MI)

---

## Project Structure

```
BCI-Challenge/
├── src/bci/
│   ├── main_offline_*.py              # Offline training + LOSO CV
│   │   ├── main_offline_Baseline.py       # CSP&BP + LDA/SVM/LR/RF
│   │   ├── main_offline_Baseline_physionet_to_target.py  # PhysioNet training then testing on target (performed worse then LOSO on target directly)
│   │   ├── main_offline_Riemann.py        # Riemannian geometry
│   │   ├── main_offline_MIRepNet.py       # MIRepNet foundation model
│   │   ├── main_offline_AdaptiveLDA.py    # Combined Adaptive LDA
│   │   └── main_offline_Advanced.py       # Deep Learning baselines
│   │
│   ├── main_online_*.py               # Online real-time inference
│   │   ├── main_online_Baseline.py
│   │   ├── main_online_Riemann.py
│   │   ├── main_online_MIRepNet.py
│   │   ├── main_online_MIRepNet_adaptive.py
│   │   ├── main_online_Advanced.py
│   │   └── main_online_AdaptiveLDA.py
│   │
│   ├── models/                        # Standardized Model Wrappers with the same API: fit(), predict(), predict_proba(), save()/load()
│   │   ├── Baseline.py                # CSP + bandpower + LDA/SVM/LR/RF
│   │   ├── riemann.py                 # Riemannian FgMDM with adaptive recentering
│   │   ├── MIRepNet.py                # MIRepNet numpy wrapper
│   │   ├── AdaptiveLDA.py             # Adaptive LDA with online parameter updates
│   │   ├── Advanced_Baselines.py      # Deep Learning Baselines Wrapper
│   │   ├── MIRepNet/                  # MIRepNet core (architectures + weights)
│   │   │   └── model/                 # EEGNet, ShallowConvNet, DeepConvNet, etc.
│   │   └── adaptive_lda_modules/      # Combined Adaptive LDA components
│   │
│   ├── preprocessing/
│   │   ├── filters.py                 # Bandpass filtering (offline & stateful online)
│   │   ├── windows.py                 # Overlapping window extraction
│   │   ├── artefact_removal.py        # Multi-criteria artifact rejection
│   │   └── preprocessing.py           # Full preprocessing pipeline
│   │
│   ├── evaluation/
│   │   └── metrics.py                 # Accuracy, F1, ECE, Brier, ITR, latency
│   │
│   ├── loading/
│   │   └── loading.py                 # Data loaders (PhysioNet + target subject)
│   │
│   ├── transfer/
│   │   └── transfer.py                # BCI controller (smoothing + UDP dispatch)
│   │
│   ├── online_visualization.py        # Real-time EEG + probability display (PyQt5)
│   └── replay.py                      # LSL stream replay for testing without hardware
│
├── resources/
│   ├── configs/bci_config.yaml        # Main configuration
│   ├── models/                        # Saved trained models (.pkl)
│   └── game_assets/dino/             # Chrome Dino game assets
│
├── pyproject.toml                     # Dependencies (uv package manager)
└── uv.lock                           # Reproducible dependency lockfile
```

---

## Getting Started

### Prerequisites

- **Python** 3.12
- **[uv](https://docs.astral.sh/uv/getting-started/)** — modern Python package manager

### Installation

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/Nikitaz1410/BCI-Challenge.git
cd BCI-Challenge

# 3. Sync dependencies
uv sync
```

### Data

Place your data under the project root as follows:

| Purpose | Path | Contents |
|--------|------|----------|
| **Target subject (raw)** | `data/eeg/sub/` | Raw `.xdf` recordings from the target subject. The folder name (`sub`) must match `target` in `bci_config.yaml`. |
| **Target subject (processed)** | `data/datasets/sub/` | Processed `.fif` and events; created automatically from the raw XDF in `data/eeg/sub/` on first run. |
| **Physionet MI** | `data/datasets/physionet/` | Downloaded automatically by the loader when using Physionet (e.g. for replay with `replay_subject_id: "Phy-###"` or for pre-training). |

You only need to provide the raw XDF files in `data/eeg/sub/` (or the folder name set by `target` in the config). All other dataset directories are created and filled by the scripts as needed.



### Offline Training

Train and evaluate models with LOSO cross-validation:

```bash
uv run src/bci/main_offline_Baseline.py       # CSP + LDA/SVM/LR/RF
uv run src/bci/main_offline_Riemann.py        # Riemannian classifier
uv run src/bci/main_offline_MIRepNet.py       # MIRepNet foundation model
uv run src/bci/main_offline_AdaptiveLDA.py    # Combined Adaptive LDA
```

> Adapt `resources/configs/bci_config.yaml` before running. First run may take longer while PhysioNet data is downloaded.

### Online — Without Dino Game

Test the real-time pipeline with pre-recorded data:

```bash
# 1. Set online: "prerecorded" in bci_config.yaml
# 2. Start the LSL data replay
uv run src/bci/replay.py

# 3. In a separate terminal, run the online decoder
uv run src/bci/main_online_Riemann.py       # or any main_online_*.py
```

### Online — With Dino Game

```bash
# 1. Set online: "dino" in bci_config.yaml
# 2. Start the Dino game (Chrome Dino Game implementation not included)

# 3. Start the EEG stream (replay or real hardware)
uv run src/bci/replay.py

# 4. In a separate terminal, run the online decoder
uv run src/bci/main_online_Riemann.py       # or any main_online_*.py
```

> Press `Ctrl+C` at any time to stop. Intermediate results are saved automatically.

---

## Configuration

All parameters are centralized in `resources/configs/bci_config.yaml`:

```yaml
# Signal Processing
window_size: 250        # Window length in samples (1.0 s at 250 Hz)
step_size: 32           # Step size in samples (~128 ms)
fs: 250                 # Sampling frequency (Hz)
frequencies: [8, 30]    # Bandpass filter range (Hz)
order: 8                # Butterworth filter order

# Artifact Removal
artefact_removal: "ar"  # "ar" (rejection) or "asr" (subspace reconstruction)

# Channel Setup
channels: [Fp1, Fp2, F3, Fz, F4, T7, C3, Cz, C4, T8, P3, Pz, P4, PO7, PO8, Oz]
remove_channels: [Fp1, Fp2, T7, T8, Oz]

# Cross-Validation
n_folds: 3              # Fallback CV folds (LOSO uses session count instead)
random_state: 42

# Online Mode
online: "dino"                    # "dino" or "prerecorded"
classification_threshold: 0.4    # Confidence threshold for sending commands
classification_buffer: 10        # Smoothing buffer size
ip: "127.0.0.1"                  # UDP target IP (Dino game)
port: 5005                       # UDP target port
```

---

## Real-Time Visualization

A PyQt5 + PyQtGraph visualization tool (`online_visualization.py`) provides:

- **Multi-channel EEG traces** — scrolling raw or filtered signals with per-channel toggle
- **Class probability bars** — live bar chart of Rest / Left / Right probabilities
- **Threshold indicator** — visual line for the classification decision boundary
- **Auto-scaling** — automatic gain adjustment

Data is streamed from the online decoder via ZMQ (port 5556).

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `mne`, `moabb` | EEG data handling and benchmarking |
| `scikit-learn` | Classifiers (LDA, SVM, LR, RF), cross-validation, metrics |
| `scipy` | Signal processing (Butterworth filter, Welch PSD) |
| `numpy`, `pandas` | Numerical computation and data management |
| `pylsl`, `pyxdf` | Lab Streaming Layer (real-time EEG acquisition) |
| `torch`, `lightning` | Deep learning models (MIRepNet, ShallowConvNet, etc.) |
| `pyriemann` | Riemannian geometry classifiers (FgMDM) |
| `autoreject` | Automated EEG artifact rejection |
| `pyqt5`, `pyqtgraph` | Real-time visualization |
| `pyzmq` | Inter-process data streaming |
| `pygame` | Dino game rendering |

Full dependency list with versions in `pyproject.toml`. Install with `uv sync`.

---

## Acknowledgements

Many thanks to **Delfina Taskin Espinoza** and **Moru Liu** for organizing and instructing this course (*Praktikum: Developing Reliable Decoders for a Brain-Computer Interface*) and closely supporting our project.

---

## References

[1] Schalk, G. et al. "BCI2000: a general-purpose brain-computer interface (BCI) system." *IEEE Trans. Biomed. Eng.* 51.6 (2004): 1034–1043.

[2] Jas, M. et al. "Autoreject: Automated artifact rejection for MEG and EEG data." *NeuroImage* 159 (2017): 417–429.

[3] Kumar, S., Yger, F. & Lotte, F. "Towards adaptive classification using Riemannian geometry approaches in brain-computer interfaces." *7th Int. Winter Conf. on BCI*, IEEE, 2019.

[4] Tao, W. et al. "ADFCNN: Attention-Based Dual-Scale Fusion CNN for MI-BCI." *IEEE Trans. NSRE* 32 (2023): 154–165.

[5] Song, Y. et al. "EEG Conformer: Convolutional Transformer for EEG Decoding." *IEEE Trans. NSRE* 31 (2022): 710–719.

[6] Mane, R. et al. "FBCNet: A Multi-view CNN for Brain-Computer Interface." arXiv:2104.01233.

[7] Schirrmeister, R.T. et al. "Deep learning with CNNs for EEG decoding and visualization." *Human Brain Mapping* 38.11 (2017): 5391–5420.

[8] Wang, T. et al. "A shallow convolutional neural network for classifying MI-EEG." *Chinese Automation Congress*, IEEE, 2019.

[9] Wang, J. et al. "IFNet: An Interactive Frequency CNN for Enhancing Motor Imagery Decoding." *IEEE Trans. NSRE* 31 (2023): 1900–1911.

[10] Lawhern, V.J. et al. "EEGNet: a compact CNN for EEG-based brain-computer interfaces." *J. Neural Eng.* 15.5 (2018): 056013.

[11] Liu, D. et al. "MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor Imagery Classification." arXiv:2507.20254.

---

## License

MIT License

Copyright (c) 2026 Nikita Zubairov, Iustin Curcean, Amal Ben Slimen, Daria Bondareva

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This project was developed as part of a university practical course at the Technical University of Munich (WiSe 2025/26).

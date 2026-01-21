# Adaptive LDA Classifier

Implementation of the Adaptive LDA classifier from:
**Wu et al. (2024)** "Adaptive LDA Classifier Enhances Real-Time Control of an EEG Brain–Computer Interface for Decoding Imagined Syllables"
*Brain Sciences*, 14(3), 196.

## Overview

This model uses Linear Discriminant Analysis (LDA) with online adaptation to handle non-stationary EEG signals during real-time BCI control.

## Key Features

- **PSD Feature Extraction**: Extracts Power Spectral Density features (8-30 Hz, 2 Hz steps)
  - **Mu band** (8-13 Hz): Motor imagery ERD/ERS
  - **Beta band** (13-30 Hz): Motor execution/imagery
- **Adaptive Updates**: Updates class means and covariance matrix online
- **Efficient Computation**: Uses Woodbury matrix identity to avoid repeated matrix inversions

**Note**: Original paper used 1-70 Hz for speech imagery. Adapted to 8-30 Hz for motor imagery (left/right hand).

## Module Structure

- `feature_extraction.py`: PSD feature extraction using Welch method
- `lda_core.py`: Standard LDA classifier implementation
- `adaptive_update.py`: Online adaptation mechanisms (Equations 4-11 from paper)

## Usage

### Configuration (bci_config.yaml):
```yaml
model: "adaptivelda"
fs: 160.0  # Sampling frequency
frequencies:  # Frequency range for feature extraction
  - 8.0   # Min freq (Mu band start)
  - 30.0  # Max freq (Beta band end)
```

### In your code:
```python
from bci.Models.AdaptiveLDA import AdaptiveLDA

# With config parameters (automatically reads from bci_config.yaml)
model_args = {
    "sfreq": config.fs,  # 160.0 Hz
    "freq_range": tuple(config.frequencies),  # (8.0, 30.0)
}
model = AdaptiveLDA(**model_args)

# Or using choose_model helper
model_args = {
    "sfreq": config.fs,
    "freq_range": tuple(config.frequencies)
}
clf = choose_model("adaptivelda", model_args)

# Train on offline data
clf.fit(signals, labels)

# Predict on new data
predictions = clf.predict(new_signals)
probabilities = clf.predict_proba(new_signals)

# Update during online control (adaptive part!)
clf.update(single_trial, true_label)
```

## Parameters

- **uc_mu** (float): Update coefficient for class means (default: 0.00625)
  - Lower values = slower adaptation
  - Optimized value from paper: 0.4 × 2^(-6)

- **uc_sigma** (float): Update coefficient for covariance (default: 0.05)
  - Lower values = slower adaptation
  - Optimized value from paper: 0.4 × 2^(-3)

## Markers

Compatible with the standard motor imagery markers:
- 0: unknown
- 1: rest
- 2: left_hand
- 3: right_hand

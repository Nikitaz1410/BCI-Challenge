# Adaptive LDA for Motor Imagery BCI

## Overview

This implementation combines two LDA classifiers with online adaptation for real-time EEG-based motor imagery classification (Rest / Left Hand / Right Hand).

**Reference:** Wu et al., "Adaptive LDA Classifier Enhances Real-Time Control of an EEG Brain–Computer Interface for Imagined-Syllables Decoding"

---

## Architecture

```
                    ┌─────────────────┐
                    │  EEG Features   │
                    │ (log-bandpower) │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
      ┌───────────────┐            ┌───────────────┐
      │   HybridLDA   │            │   Core LDA    │
      │  (2-stage)    │            │  (3-class)    │
      └───────┬───────┘            └───────┬───────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                   ┌─────────────────┐
                   │ Confidence-based│
                   │    Selection    │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Prediction    │
                   │   + Adaptation  │
                   └─────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **HybridLDA** | 2-stage hierarchical: Stage A (Rest vs Move) → Stage B (Left vs Right) |
| **Core LDA** | Standard 3-class LDA with regularization |
| **Adaptive Selection** | High confidence → Core LDA; Low confidence → HybridLDA |

---

## Key Features

1. **Online Adaptation**: Class means updated via exponential moving average after each trial
2. **Confidence-based Selection**: Switches between models based on prediction confidence
3. **Adaptive Learning Rate**: Lower confidence → higher learning rate
4. **Ensemble Predictions**: Weighted combination when beneficial

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.7 | Model selection threshold |
| `ensemble_weight` | 0.5 | Ensemble weighting (0=single, 1=equal) |
| `move_threshold` | 0.6 | Stage A threshold for movement detection |
| `reg` | 1e-3 | Covariance regularization |
| `shrinkage_alpha` | 0.05 | Shrinkage parameter |
| `uc_mu` | 0.003125 | Mean update rate (0.4 × 2⁻⁷) |

---

## Usage

### Offline Training
```bash
python src/bci/main_offline_AdaptiveLDA.py
```
- Performs 11-fold cross-validation (session-wise grouping)
- Saves model to `resources/models/combined_adaptive_lda.pkl`

### Online Classification
```bash
python src/bci/main_online_AdaptiveLDA.py
```
- Requires LSL streams (EEG + Markers)
- Adapts in real-time after each trial

---

## Feature Extraction

**Log-bandpower features** extracted from:
- **Mu band**: 8-12 Hz (sensorimotor rhythm)
- **Beta band**: 13-30 Hz (motor-related activity)

For each channel: `feature = log(bandpower)`

---

## Markers

| Code | Class |
|------|-------|
| 0 | Rest |
| 1 | Left Hand |
| 2 | Right Hand |

---

## Files

```
adaptive_lda_modules/
├── combined_adaptive_lda.py   # Main classifier (CombinedAdaptiveLDA)
├── hybrid_lda.py              # 2-stage hierarchical LDA
├── lda_core.py                # Standard 3-class LDA
├── feature_extraction.py      # Log-bandpower extraction
└── README.md                  # This file
```

---

## Performance

Cross-validation results are displayed after running `main_offline_AdaptiveLDA.py`:
- Accuracy, Balanced Accuracy, F1 Score
- ECE (Expected Calibration Error), Brier Score
- Confusion matrix saved as PNG

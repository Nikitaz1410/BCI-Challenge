# Online Replay Mode Guide

This guide explains how to use the online replay mode for evaluating static vs adaptive LDA performance.

## Overview

The replay mode (`main_online_replay.py`) simulates the online pipeline on recorded EEG sessions, allowing you to:
- Evaluate model performance on recorded data
- Compare static vs adaptive LDA using A/B testing
- Generate detailed metrics and logs

## Files Created/Modified

**Created:**
- `src/bci/main_online_replay.py` - Main replay script
- `src/bci/utils/create_replay_npz.py` - Helper to convert XDF to NPZ
- `REPLAY_MODE_GUIDE.md` - This guide

**No modifications** to existing online script (`main_online_AdaptiveLDA.py`) - live mode remains intact.

## Input Formats

### NPZ Format (Recommended)

NPZ files should contain:
- `eeg`: Shape `(n_samples, n_channels)` or `(n_channels, n_samples)`
- `eeg_timestamps`: Shape `(n_samples,)` - timestamps in seconds
- `ch_names`: List/array of channel names (length `n_channels`)
- `markers`: Array of marker values (integers: 1=rest, 2=left_hand, 3=right_hand)
- `marker_timestamps`: Shape `(n_markers,)` - marker timestamps in seconds

### XDF Format

XDF files are supported using the existing `pyxdf` parser. The script expects:
- An EEG stream named "EEG"
- A marker stream named "MyDinoGameMarkerStream", "Markers", or "Labels_Stream"

## Usage

### Basic Replay (Single Mode)

Run replay with static or adaptive mode based on config:

```bash
# Using NPZ file
python -m src.bci.main_online_replay --replay_npz path/to/replay.npz

# Using XDF file
python -m src.bci.main_online_replay --replay_xdf path/to/replay.xdf
```

The script will use `adaptation.enabled` from config to determine mode.

### A/B Test Mode

Compare static vs adaptive on the same data:

```bash
# Run A/B test
python -m src.bci.main_online_replay --replay_npz path/to/replay.npz --ab_test true

# Or with XDF
python -m src.bci.main_online_replay --replay_xdf path/to/replay.xdf --ab_test true
```

### Specify Output Directory

```bash
python -m src.bci.main_online_replay --replay_npz replay.npz --output_dir results/
```

## Converting XDF to NPZ

Use the helper script to convert XDF files:

```bash
python -m src.bci.utils.create_replay_npz --xdf path/to/file.xdf --output replay.npz
```

## Metrics

### Window-Level Metrics

For each sliding window within a trial:
- `y_pred`: Predicted class
- `proba_true`: Probability of true class
- `is_correct`: Whether prediction matches true label

### Trial-Level Metrics

For each completed trial:
- `trial_idx`: Trial index
- `y_true`: True class label (0=rest, 1=left, 2=right)
- `trial_mean_proba_true`: Mean probability of true class across windows
- `trial_window_accuracy`: Mean accuracy across windows in trial
- `trial_pred_majority`: Most frequent predicted class
- `adapted`: Whether model was adapted (True/False)

### End-of-Run Summary

- Total number of trials
- Overall mean trial_window_accuracy
- Confusion matrix (trial_pred_majority vs y_true)
- Per-class accuracy
- Bias metric: max(acc_by_class) - min(acc_by_class)

### A/B Test Comparison

When `--ab_test true`:
- Delta mean trial_window_accuracy (adaptive - static)
- Delta bias (static_bias - adaptive_bias)
- Per-class accuracy comparison

## Output Files

Log files are saved as JSON in the output directory (default: current directory):

- `replay_static_<filename>.json` - Static mode results
- `replay_adaptive_<filename>.json` - Adaptive mode results

Each JSON contains:
- `mode`: "static" or "adaptive"
- `adaptation_enabled`: Boolean
- `n_trials`: Number of completed trials
- `trial_metrics`: List of trial-level metrics
- `window_metrics`: List of window-level metrics
- `summary`: End-of-run summary statistics

## Example Workflow

1. **Convert XDF to NPZ** (if needed):
   ```bash
   python -m src.bci.utils.create_replay_npz --xdf data/session.xdf --output replay.npz
   ```

2. **Run A/B test**:
   ```bash
   python -m src.bci.main_online_replay --replay_npz replay.npz --ab_test true --output_dir results/
   ```

3. **Analyze results**:
   - Check console output for summary
   - Load JSON files for detailed analysis
   - Compare static vs adaptive performance

## Notes

- The replay mode uses the **same pipeline** as live mode (filter, CAR, channel selection)
- Trial state machine matches live logic exactly
- Updates happen **only at trial end** with known true labels
- Never updates per sliding window
- Channel selection enforces standard 16-channel order

## Troubleshooting

**Error: "Channel X not found in recording"**
- Ensure your recording has all 16 standard channels
- Check channel names match standard order

**Error: "No trials completed"**
- Check marker format (should be 1, 2, 3 for rest, left, right)
- Verify marker timestamps align with EEG timestamps

**Error: "Model not found"**
- Train model first using `main_offline_AdaptiveLDA.ipynb`
- Ensure model is saved to `resources/models/adaptivelda_model.pkl`

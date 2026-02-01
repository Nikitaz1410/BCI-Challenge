# Numerical Stability + Final Validation Guide

This guide documents the numerical stability hardening and final validation features added to the Adaptive LDA implementation.

## Overview

The implementation now includes:
- **Numerical stability hardening** with post-update validation and fallback mechanisms
- **Update counters** for debugging and auditing
- **Paper-grade validation plots** comparing static vs adaptive performance
- **Final validation script** for comprehensive testing

## Numerical Stability Features

### Post-Update Validation

After every adaptive update, the system checks:
1. **No NaNs or infs** in:
   - `class_means_`
   - `cov_`
   - `inv_cov_`

2. **Covariance symmetry**:
   - Enforces `cov = 0.5 * (cov + cov.T)`
   - Warns if asymmetry detected

3. **Fallback mechanism**:
   - If validation fails, attempts regularized inversion: `cov_reg = cov + eps * I`
   - Falls back to pseudo-inverse if inversion fails
   - Raises `RuntimeError` if fallback also fails (does not silently continue)

### Regularization Consistency

- `eps * I` is added:
  - After covariance update
  - After fallback inversion
- `eps` is configurable (default: `1e-6`)

### UC Parameter Safety

- `uc_mu` and `uc_sigma` must be in `(0, 1)` (strict)
- Raises `ValueError` if invalid
- Clamped to safe range `(1e-10, 1-1e-10)` if needed

### Update Counters

The `AdaptiveLDA` class now tracks:
- `n_updates_`: Total number of updates
- `n_updates_per_class_`: Dict mapping class index to update count

Access via `get_update_stats()`:
```python
stats = clf.get_update_stats()
print(f"Total updates: {stats['n_updates']}")
print(f"Updates per class: {stats['n_updates_per_class']}")
```

## Validation Plots

The plotting script (`plot_adaptive_vs_static.py`) generates three paper-style plots:

1. **Accuracy vs Trial Index**
   - Shows static and adaptive accuracy over trials
   - Includes mean lines for comparison

2. **Mean True-Class Probability vs Trial Index**
   - Shows confidence in true class predictions
   - Helps identify when adaptive mode improves confidence

3. **Bias vs Trial Index (Cumulative)**
   - Shows bias metric: `max(class_acc) - min(class_acc)`
   - Lower bias = more balanced performance across classes
   - Positive delta = adaptive reduced bias

## Usage

### Final Replay Validation

Run the complete validation pipeline:

```bash
python -m src.bci.Evaluation.run_final_replay_validation --replay_npz path/to/replay.npz
```

This will:
1. Run replay in static mode
2. Run replay in adaptive mode
3. Generate validation plots
4. Print summary with accuracy and bias metrics

### Generate Plots Only

If you already have log files:

```bash
python -m src.bci.Evaluation.plot_adaptive_vs_static \
    --static_log path/to/replay_static_*.json \
    --adaptive_log path/to/replay_adaptive_*.json
```

### Custom Output Directory

```bash
python -m src.bci.Evaluation.run_final_replay_validation \
    --replay_npz replay.npz \
    --output_dir validation_results/
```

## Output Files

**Logs** (saved to `output_dir/logs/`):
- `replay_static_<filename>.json`
- `replay_adaptive_<filename>.json`

**Plots** (saved to `resources/plots/`):
- `accuracy_vs_trial.png`
- `probability_vs_trial.png`
- `bias_vs_trial.png`

## Validation Summary

The final validation script prints:
- **Static Accuracy**: Overall mean trial window accuracy
- **Adaptive Accuracy**: Overall mean trial window accuracy
- **Delta Accuracy**: `adaptive - static` (positive = adaptive better)
- **Static Bias**: Bias metric for static mode
- **Adaptive Bias**: Bias metric for adaptive mode
- **Bias Reduction**: `static_bias - adaptive_bias` (positive = adaptive reduced bias)
- **Total Trials**: Number of completed trials

## Files Modified/Created

**Modified:**
- `src/bci/Models/AdaptiveLDA_modules/adaptive_update.py` - Numerical stability hardening
- `src/bci/Models/AdaptiveLDA.py` - Update counters and validation

**Created:**
- `src/bci/Evaluation/plot_adaptive_vs_static.py` - Validation plots
- `src/bci/Evaluation/run_final_replay_validation.py` - Final validation script
- `NUMERICAL_STABILITY_GUIDE.md` - This guide

## Paper-Grade Verification

The implementation is now "paper-grade" with:
- ✅ Numerical stability safeguards
- ✅ Auditable update tracking
- ✅ Comprehensive validation plots
- ✅ Automated validation pipeline
- ✅ Clear error handling and fallbacks

Run the final validation script to verify the paper implementation works correctly.

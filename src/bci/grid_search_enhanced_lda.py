"""
Grid Search for EnhancedAdaptiveLDA Enhancement Combinations.

Tests all 32 combinations of the 5 enhancement features:
1. use_adaptive_selection: Switches between standard 3-class LDA (when confident) 
   and HybridLDA (when uncertain) based on prediction confidence
2. use_improved_composition: Uses confidence-weighted Stage B probabilities 
   instead of simple thresholding for better probability estimates
3. use_temporal_smoothing: Applies majority/weighted voting over a sliding window 
   of recent predictions to reduce noise and improve stability
4. use_dynamic_threshold: Automatically adjusts move_threshold based on recent 
   prediction performance (increases if too many false positives, decreases if too conservative)
5. use_adaptive_learning_rate: Adjusts the update coefficient (uc_mu) based on 
   prediction correctness and confidence (learns faster when wrong, slower when confident)

Reports which combinations give the best CV accuracy.
"""

import itertools
import pickle
import sys
import time
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Evaluation
from bci.Evaluation.metrics import compile_metrics

# Data Acquisition
from bci.loading.loading import (
    load_target_subject_data,
    create_subject_train_set,
    create_subject_test_set
)

# Preprocessing
from bci.Preprocessing.filters import Filter
from bci.Preprocessing.windows import epochs_to_windows, epochs_windows_from_fold

# Models - Enhanced Adaptive LDA
from bci.Models.AdaptiveLDA_modules.enhanced_adaptive_lda import EnhancedAdaptiveLDA
from bci.Models.AdaptiveLDA_modules.feature_extraction import extract_log_bandpower_features

# Utils
from bci.utils.bci_config import load_config


def extract_features(signals, sfreq):
    """Extract log-bandpower features."""
    return extract_log_bandpower_features(signals, sfreq=sfreq, mu_band=(8, 12), beta_band=(13, 30))


def run_cv_with_config(
    combined_epochs,
    groups,
    config,
    use_adaptive_selection,        # Switch between standard LDA (confident) and HybridLDA (uncertain)
    use_improved_composition,      # Use confidence-weighted Stage B probabilities
    use_temporal_smoothing,        # Apply majority/weighted voting over recent predictions
    use_dynamic_threshold,         # Automatically adjust move_threshold based on performance
    use_adaptive_learning_rate,    # Adjust uc_mu based on prediction correctness/confidence
    n_folds=None
):
    """
    Run cross-validation with a specific configuration of enhancement features.

    Parameters:
    -----------
    use_adaptive_selection : bool
        Switches between standard 3-class LDA (when confident) and HybridLDA (when uncertain)
    use_improved_composition : bool
        Uses confidence-weighted Stage B probabilities instead of simple thresholding
    use_temporal_smoothing : bool
        Applies majority/weighted voting over a sliding window of recent predictions
    use_dynamic_threshold : bool
        Automatically adjusts move_threshold based on recent prediction performance
    use_adaptive_learning_rate : bool
        Adjusts the update coefficient (uc_mu) based on prediction correctness and confidence

    Returns:
        dict with 'mean_accuracy', 'std_accuracy', 'fold_accuracies'
    """
    X_train = combined_epochs.get_data()
    y_train = combined_epochs.events[:, 2]

    # Determine number of folds
    unique_groups = np.unique(groups)
    if n_folds is None:
        n_folds = min(len(unique_groups), config.n_folds)
    n_folds = min(n_folds, len(unique_groups))

    if n_folds < 2:
        return {'mean_accuracy': np.nan, 'std_accuracy': np.nan, 'fold_accuracies': []}

    gkf = GroupKFold(n_splits=n_folds)
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
        # Window the data for this fold
        fold_windowed = epochs_windows_from_fold(
            combined_epochs,
            groups,
            train_idx,
            val_idx,
            window_size=config.window_size,
            step_size=config.step_size,
        )

        X_train_fold = fold_windowed["X_train"]
        y_train_fold = fold_windowed["y_train"]
        X_val_fold = fold_windowed["X_val"]
        y_val_fold = fold_windowed["y_val"]

        # Extract features
        train_features = extract_features(X_train_fold, config.fs)
        val_features = extract_features(X_val_fold, config.fs)

        # Train Enhanced Adaptive LDA with specified configuration
        # Enhancement features:
        # - use_adaptive_selection: Switch between standard LDA (confident) and HybridLDA (uncertain)
        # - use_improved_composition: Use confidence-weighted Stage B probabilities
        # - use_temporal_smoothing: Apply majority/weighted voting over recent predictions
        # - use_dynamic_threshold: Automatically adjust move_threshold based on performance
        # - use_adaptive_learning_rate: Adjust uc_mu based on prediction correctness/confidence
        fold_clf = EnhancedAdaptiveLDA(
            move_threshold=0.5,
            reg=1e-2,
            shrinkage_alpha=0.1,
            uc_mu=0.4 * 2**-6,
            use_adaptive_selection=use_adaptive_selection,
            use_improved_composition=use_improved_composition,
            use_temporal_smoothing=use_temporal_smoothing,
            use_dynamic_threshold=use_dynamic_threshold,
            use_adaptive_learning_rate=use_adaptive_learning_rate,
        )

        fold_clf.fit(train_features, y_train_fold)

        # Predict on validation fold
        val_preds = fold_clf.predict(val_features)

        # Compute accuracy
        fold_acc = np.mean(val_preds == y_val_fold)
        fold_accuracies.append(fold_acc)

    return {
        'mean_accuracy': np.mean(fold_accuracies),
        'std_accuracy': np.std(fold_accuracies),
        'fold_accuracies': fold_accuracies
    }


def run_grid_search():
    """Run grid search over all enhancement combinations."""

    # ==========================================================================
    # 1. Setup and Load Data
    # ==========================================================================
    script_dir = Path(__file__).parent.parent.parent

    if (script_dir / "src").exists() and (script_dir / "data").exists():
        current_wd = script_dir
    elif (script_dir / "BCI-Challenge" / "src").exists():
        current_wd = script_dir / "BCI-Challenge"
    else:
        current_wd = script_dir

    config_path = current_wd / "resources" / "configs" / "bci_config.yaml"
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    np.random.seed(config.random_state)

    filter_obj = Filter(config, online=False)

    test_data_source_path = current_wd / "data" / "eeg" / config.target
    test_data_target_path = current_wd / "data" / "datasets" / config.target

    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    all_target_raws, all_target_events, target_event_id, target_sub_ids, target_metadata = load_target_subject_data(
        root=current_wd,
        source_path=test_data_source_path,
        target_path=test_data_target_path,
        resample=None,
    )
    print(f"Loaded {len(all_target_raws)} sessions from target subject data.")

    # Create training set
    x_raw_train, events_train, train_filenames, sub_ids_train, train_indices = create_subject_train_set(
        config,
        all_target_raws,
        all_target_events,
        target_metadata["filenames"],
        num_general=4,
        num_dino=9,
        num_supression=0,
        shuffle=True
    )
    print(f"Created training set with {len(x_raw_train)} files.")

    # ==========================================================================
    # 2. Preprocess Data
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)

    all_epochs_list = []
    for raw, events, sub_id, filename in zip(x_raw_train, events_train, sub_ids_train, train_filenames):
        filtered_raw = raw.copy()
        filtered_raw.apply_function(filter_obj.apply_filter_offline)
        filtered_raw.drop_channels(config.remove_channels)

        epochs = mne.Epochs(
            filtered_raw,
            events,
            event_id=target_event_id,
            tmin=0.3,
            tmax=4.0,
            preload=True,
            baseline=None,
        )

        metadata = pd.DataFrame({
            "subject_id": [sub_id] * len(epochs),
            "filename": [filename] * len(epochs),
            "condition": epochs.events[:, 2],
        })
        epochs.metadata = metadata
        all_epochs_list.append(epochs)

    combined_epochs = mne.concatenate_epochs(all_epochs_list)
    groups = combined_epochs.metadata["filename"].values

    print(f"Total epochs: {len(combined_epochs)}")
    print(f"Unique groups (files): {len(np.unique(groups))}")

    # ==========================================================================
    # 3. Grid Search Over All Combinations
    # ==========================================================================
    print("\n" + "=" * 60)
    print("GRID SEARCH: TESTING ALL 32 ENHANCEMENT COMBINATIONS")
    print("=" * 60)

    # Define all enhancement features with explanations:
    # 1. use_adaptive_selection: Switches between standard 3-class LDA (when confident) 
    #    and HybridLDA (when uncertain) based on prediction confidence
    # 2. use_improved_composition: Uses confidence-weighted Stage B probabilities 
    #    instead of simple thresholding for better probability estimates
    # 3. use_temporal_smoothing: Applies majority/weighted voting over a sliding window 
    #    of recent predictions to reduce noise and improve stability
    # 4. use_dynamic_threshold: Automatically adjusts move_threshold based on recent 
    #    prediction performance (increases if too many false positives, decreases if too conservative)
    # 5. use_adaptive_learning_rate: Adjusts the update coefficient (uc_mu) based on 
    #    prediction correctness and confidence (learns faster when wrong, slower when confident)
    features = [
        'use_adaptive_selection',
        'use_improved_composition',
        'use_temporal_smoothing',
        'use_dynamic_threshold',
        'use_adaptive_learning_rate'
    ]

    # Generate all 32 combinations (2^5)
    all_combinations = list(itertools.product([False, True], repeat=5))

    results = []

    for i, combo in enumerate(all_combinations):
        # Unpack the 5 enhancement feature flags:
        # [0] use_adaptive_selection: Switch between standard LDA (confident) and HybridLDA (uncertain)
        # [1] use_improved_composition: Use confidence-weighted Stage B probabilities
        # [2] use_temporal_smoothing: Apply majority/weighted voting over recent predictions
        # [3] use_dynamic_threshold: Automatically adjust move_threshold based on performance
        # [4] use_adaptive_learning_rate: Adjust uc_mu based on prediction correctness/confidence
        use_adaptive_selection = combo[0]
        use_improved_composition = combo[1]
        use_temporal_smoothing = combo[2]
        use_dynamic_threshold = combo[3]
        use_adaptive_learning_rate = combo[4]

        # Create readable config string
        config_str = []
        if use_adaptive_selection: config_str.append("AdaptSel")
        if use_improved_composition: config_str.append("ImpComp")
        if use_temporal_smoothing: config_str.append("TempSmooth")
        if use_dynamic_threshold: config_str.append("DynThresh")
        if use_adaptive_learning_rate: config_str.append("AdaptLR")
        config_name = "+".join(config_str) if config_str else "Baseline (all off)"

        print(f"\n[{i+1}/32] Testing: {config_name}")

        start_time = time.time()

        cv_results = run_cv_with_config(
            combined_epochs=combined_epochs,
            groups=groups,
            config=config,
            use_adaptive_selection=use_adaptive_selection,
            use_improved_composition=use_improved_composition,
            use_temporal_smoothing=use_temporal_smoothing,
            use_dynamic_threshold=use_dynamic_threshold,
            use_adaptive_learning_rate=use_adaptive_learning_rate,
        )

        elapsed = time.time() - start_time

        result = {
            'config_name': config_name,
            'use_adaptive_selection': use_adaptive_selection,
            'use_improved_composition': use_improved_composition,
            'use_temporal_smoothing': use_temporal_smoothing,
            'use_dynamic_threshold': use_dynamic_threshold,
            'use_adaptive_learning_rate': use_adaptive_learning_rate,
            'mean_accuracy': cv_results['mean_accuracy'],
            'std_accuracy': cv_results['std_accuracy'],
            'fold_accuracies': cv_results['fold_accuracies'],
            'elapsed_time': elapsed
        }
        results.append(result)

        print(f"   Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f} ({elapsed:.1f}s)")

    # ==========================================================================
    # 4. Analyze and Display Results
    # ==========================================================================
    print("\n" + "=" * 60)
    print("GRID SEARCH RESULTS (SORTED BY ACCURACY)")
    print("=" * 60)

    # Sort by mean accuracy (descending)
    results_sorted = sorted(results, key=lambda x: x['mean_accuracy'], reverse=True)

    # Create results DataFrame
    df_results = pd.DataFrame([{
        'Config': r['config_name'],
        'Accuracy': f"{r['mean_accuracy']:.4f}",
        'Std': f"{r['std_accuracy']:.4f}",
        'AdaptSel': r['use_adaptive_selection'],
        'ImpComp': r['use_improved_composition'],
        'TempSmooth': r['use_temporal_smoothing'],
        'DynThresh': r['use_dynamic_threshold'],
        'AdaptLR': r['use_adaptive_learning_rate'],
    } for r in results_sorted])

    print("\nTop 10 Configurations:")
    print("-" * 80)
    for i, r in enumerate(results_sorted[:10]):
        print(f"{i+1:2d}. {r['config_name']:<50} | Acc: {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")

    print("\n" + "-" * 80)
    print("Bottom 5 Configurations:")
    print("-" * 80)
    for i, r in enumerate(results_sorted[-5:]):
        rank = len(results_sorted) - 4 + i
        print(f"{rank:2d}. {r['config_name']:<50} | Acc: {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")

    # ==========================================================================
    # 5. Feature Impact Analysis
    # ==========================================================================
    print("\n" + "=" * 60)
    print("FEATURE IMPACT ANALYSIS")
    print("=" * 60)

    for feature in features:
        # Compare mean accuracy when feature is ON vs OFF
        on_accs = [r['mean_accuracy'] for r in results if r[feature]]
        off_accs = [r['mean_accuracy'] for r in results if not r[feature]]

        mean_on = np.mean(on_accs)
        mean_off = np.mean(off_accs)
        diff = mean_on - mean_off

        impact = "HELPS" if diff > 0.005 else ("HURTS" if diff < -0.005 else "NEUTRAL")

        feature_short = feature.replace('use_', '').replace('_', ' ').title()
        print(f"{feature_short:<25} | ON: {mean_on:.4f} | OFF: {mean_off:.4f} | Diff: {diff:+.4f} | {impact}")

    # ==========================================================================
    # 6. Save Results
    # ==========================================================================
    results_path = current_wd / "resources" / "grid_search_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Save full results as pickle
    pickle_path = current_wd / "resources" / "grid_search_results.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results_sorted, f)
    print(f"Full results saved to: {pickle_path}")

    # ==========================================================================
    # 7. Best Configuration Summary
    # ==========================================================================
    best = results_sorted[0]
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"Name: {best['config_name']}")
    print(f"Mean Accuracy: {best['mean_accuracy']:.4f} ± {best['std_accuracy']:.4f}")
    print(f"Fold Accuracies: {[f'{a:.4f}' for a in best['fold_accuracies']]}")
    print("\nSettings:")
    print(f"  use_adaptive_selection:    {best['use_adaptive_selection']}")
    print(f"  use_improved_composition:  {best['use_improved_composition']}")
    print(f"  use_temporal_smoothing:    {best['use_temporal_smoothing']}")
    print(f"  use_dynamic_threshold:     {best['use_dynamic_threshold']}")
    print(f"  use_adaptive_learning_rate:{best['use_adaptive_learning_rate']}")

    return results_sorted, df_results


if __name__ == "__main__":
    results, df = run_grid_search()

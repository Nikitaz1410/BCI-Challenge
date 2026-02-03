"""
Quick test of individual EnhancedAdaptiveLDA features.
Tests baseline + each feature individually (6 configs total).
"""

import sys
import time
from pathlib import Path

src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from bci.loading.loading import load_target_subject_data, create_subject_train_set
from bci.Preprocessing.filters import Filter
from bci.Preprocessing.windows import epochs_windows_from_fold
from bci.Models.AdaptiveLDA_modules.enhanced_adaptive_lda import EnhancedAdaptiveLDA
from bci.Models.AdaptiveLDA_modules.feature_extraction import extract_log_bandpower_features
from bci.utils.bci_config import load_config


def extract_features(signals, sfreq):
    return extract_log_bandpower_features(signals, sfreq=sfreq, mu_band=(8, 12), beta_band=(13, 30))


def run_cv(combined_epochs, groups, config, **kwargs):
    """Run CV with specified EnhancedAdaptiveLDA settings."""
    X_train = combined_epochs.get_data()
    y_train = combined_epochs.events[:, 2]

    unique_groups = np.unique(groups)
    n_folds = min(len(unique_groups), config.n_folds)

    if n_folds < 2:
        return np.nan, np.nan, []

    gkf = GroupKFold(n_splits=n_folds)
    fold_accs = []

    for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
        fold_windowed = epochs_windows_from_fold(
            combined_epochs, groups, train_idx, val_idx,
            window_size=config.window_size, step_size=config.step_size,
        )

        train_features = extract_features(fold_windowed["X_train"], config.fs)
        val_features = extract_features(fold_windowed["X_val"], config.fs)

        clf = EnhancedAdaptiveLDA(
            move_threshold=0.5, reg=1e-2, shrinkage_alpha=0.1, uc_mu=0.4 * 2**-6,
            **kwargs
        )
        clf.fit(train_features, fold_windowed["y_train"])
        preds = clf.predict(val_features)
        fold_accs.append(np.mean(preds == fold_windowed["y_val"]))

    return np.mean(fold_accs), np.std(fold_accs), fold_accs


if __name__ == "__main__":
    # Setup
    script_dir = Path(__file__).parent.parent.parent
    if (script_dir / "src").exists() and (script_dir / "data").exists():
        current_wd = script_dir
    elif (script_dir / "BCI-Challenge" / "src").exists():
        current_wd = script_dir / "BCI-Challenge"
    else:
        current_wd = script_dir

    config = load_config(current_wd / "resources" / "configs" / "bci_config.yaml")
    np.random.seed(config.random_state)
    filter_obj = Filter(config, online=False)

    # Load data
    print("Loading data...")
    test_data_source_path = current_wd / "data" / "eeg" / config.target
    test_data_target_path = current_wd / "data" / "datasets" / config.target

    all_target_raws, all_target_events, target_event_id, target_sub_ids, target_metadata = load_target_subject_data(
        root=current_wd, source_path=test_data_source_path, target_path=test_data_target_path, resample=None,
    )

    x_raw_train, events_train, train_filenames, sub_ids_train, _ = create_subject_train_set(
        config, all_target_raws, all_target_events, target_metadata["filenames"],
        num_general=4, num_dino=9, num_supression=0, shuffle=True
    )

    # Preprocess
    print("Preprocessing...")
    all_epochs_list = []
    for raw, events, sub_id, filename in zip(x_raw_train, events_train, sub_ids_train, train_filenames):
        filtered_raw = raw.copy()
        filtered_raw.apply_function(filter_obj.apply_filter_offline)
        filtered_raw.drop_channels(config.remove_channels)
        epochs = mne.Epochs(filtered_raw, events, event_id=target_event_id,
                           tmin=0.3, tmax=4.0, preload=True, baseline=None)
        epochs.metadata = pd.DataFrame({
            "subject_id": [sub_id] * len(epochs),
            "filename": [filename] * len(epochs),
            "condition": epochs.events[:, 2],
        })
        all_epochs_list.append(epochs)

    combined_epochs = mne.concatenate_epochs(all_epochs_list)
    groups = combined_epochs.metadata["filename"].values
    print(f"Data ready: {len(combined_epochs)} epochs, {len(np.unique(groups))} groups")

    # Test configurations
    configs = [
        ("Baseline (all OFF)", dict(use_adaptive_selection=False, use_improved_composition=False,
                                    use_temporal_smoothing=False, use_dynamic_threshold=False,
                                    use_adaptive_learning_rate=False)),
        ("1. adaptive_selection ONLY", dict(use_adaptive_selection=True, use_improved_composition=False,
                                            use_temporal_smoothing=False, use_dynamic_threshold=False,
                                            use_adaptive_learning_rate=False)),
        ("2. improved_composition ONLY", dict(use_adaptive_selection=False, use_improved_composition=True,
                                              use_temporal_smoothing=False, use_dynamic_threshold=False,
                                              use_adaptive_learning_rate=False)),
        ("3. temporal_smoothing ONLY", dict(use_adaptive_selection=False, use_improved_composition=False,
                                            use_temporal_smoothing=True, use_dynamic_threshold=False,
                                            use_adaptive_learning_rate=False)),
        ("4. dynamic_threshold ONLY", dict(use_adaptive_selection=False, use_improved_composition=False,
                                           use_temporal_smoothing=False, use_dynamic_threshold=True,
                                           use_adaptive_learning_rate=False)),
        ("5. adaptive_learning_rate ONLY", dict(use_adaptive_selection=False, use_improved_composition=False,
                                                use_temporal_smoothing=False, use_dynamic_threshold=False,
                                                use_adaptive_learning_rate=True)),
    ]

    print("\n" + "=" * 70)
    print("TESTING INDIVIDUAL FEATURES")
    print("=" * 70)

    results = []
    for name, kwargs in configs:
        print(f"\nTesting: {name}")
        start = time.time()
        mean_acc, std_acc, fold_accs = run_cv(combined_epochs, groups, config, **kwargs)
        elapsed = time.time() - start
        results.append((name, mean_acc, std_acc, fold_accs))
        print(f"   Accuracy: {mean_acc:.4f} +/- {std_acc:.4f} ({elapsed:.1f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (sorted by accuracy)")
    print("=" * 70)

    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    baseline_acc = results[0][1]  # First result is baseline

    for i, (name, acc, std, _) in enumerate(results_sorted):
        diff = acc - baseline_acc
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        marker = " <-- BEST" if i == 0 else ""
        print(f"{i+1}. {name:<35} | Acc: {acc:.4f} +/- {std:.4f} | vs baseline: {diff_str}{marker}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    best_name, best_acc, best_std, _ = results_sorted[0]
    print(f"Best single feature: {best_name}")
    print(f"Accuracy: {best_acc:.4f} +/- {best_std:.4f}")

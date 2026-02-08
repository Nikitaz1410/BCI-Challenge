"""
Offline Training and Evaluation Script for Combined Adaptive LDA.

1. Loads training data using create_subject_train_set (all 23 files for CV)
2. Trains a CombinedAdaptiveLDA classifier (HybridLDA + Core LDA with adaptive selection)
3. Performs 11-fold cross-validation with session-wise splits (matching baseline)
4. Simulates offline adaptation to compare accuracy with and without adaptation
5. Saves the trained model to combined_adaptive_lda.pkl

Uses the same CV methodology as main_offline_Baseline.py for consistency.
"""

import pickle
import random
import re
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

# Evaluation
from bci.evaluation.metrics import MetricsTable, compile_metrics

# Data Acquisition
from bci.loading.loading import (
    load_target_subject_data,
    create_subject_train_set,
)

# Preprocessing
from bci.preprocessing.filters import Filter
from bci.preprocessing.windows import epochs_to_windows, epochs_windows_from_fold
from bci.preprocessing.artefact_removal import ArtefactRemoval

# Models - CombinedAdaptiveLDA (HybridLDA + Core LDA with adaptive selection)
from bci.models.adaptive_lda_modules.combined_adaptive_lda import CombinedAdaptiveLDA
from bci.models.adaptive_lda_modules.feature_extraction import extract_log_bandpower_features

# Utils
from bci.utils.bci_config import load_config


# =============================================================================
# Session ID Extraction (matching baseline)
# =============================================================================
def _session_id_from_filename(filename: str) -> str:
    """
    Extract session identifier from BIDS-style filename for CV grouping.

    E.g. "sub-P999_ses-S002_task-dino_run-001_eeg_raw" -> "sub-P999_ses-S002".
    Multiple files (runs) from the same session get the same ID so they stay
    in the same CV fold. If the pattern is not found, returns the full filename
    (one file = one group).
    """
    # Strip _raw suffix if present
    base = filename.replace("_raw", "").strip("_")
    match = re.match(r"(sub-[^_]+_ses-[^_]+)", base)
    if match:
        return match.group(1)
    return filename


# =============================================================================
# Feature Extraction Helper (shared between offline and online)
# =============================================================================
def extract_features(signals, sfreq):
    """
    Shared feature extraction function.

    Uses log-bandpower features (mu: 8-12 Hz, beta: 13-30 Hz).

    Parameters:
    -----------
    signals : np.ndarray
        Shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz

    Returns:
    --------
    features : np.ndarray
        Shape (n_trials, n_features) or (n_features,)
    """
    return extract_log_bandpower_features(signals, sfreq=sfreq, mu_band=(8, 12), beta_band=(13, 30))


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Load the config file
    # Detect project root: handle both workspace root and BCI-Challenge subdirectory
    # Script is at: [workspace]/BCI-Challenge/src/bci/main_offline_AdaptiveLDA.py
    script_dir = Path(__file__).parent.parent.parent  # Goes up 3 levels from script
    
    # Check if we're in a BCI-Challenge subdirectory (workspace structure)
    # The script is at: BCI-Challenge/src/bci/main_offline_AdaptiveLDA.py
    # So script_dir should be BCI-Challenge directory
    # Check if script_dir contains "src" and "data" directories to confirm it's the project root
    if (script_dir / "src").exists() and (script_dir / "data").exists():
        # This is the BCI-Challenge project root
        current_wd = script_dir
    elif (script_dir / "BCI-Challenge" / "src").exists() and (script_dir / "BCI-Challenge" / "data").exists():
        # We're in workspace root, need to go into BCI-Challenge
        current_wd = script_dir / "BCI-Challenge"
    else:
        # Fallback: assume script_dir is correct
        current_wd = script_dir

    try:
        config_path = current_wd / "resources" / "configs" / "bci_config.yaml"
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        print("Configuration loaded successfully!")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Initialize variables - set all seeds for reproducibility
    random.seed(config.random_state)
    np.random.seed(config.random_state)

    # Force 11-fold CV (override config, matching baseline)
    # Note: GroupKFold will be recreated after we know the actual number of groups
    n_folds_target = 11
    print(f"Target: {n_folds_target}-fold cross-validation (matching baseline)")
    print(f"Random state: {config.random_state} (for reproducibility)")
    
    gkf = None  # Will be created after we know the number of unique groups

    filter_obj = Filter(config, online=False)

    # Paths for target subject data (matching baseline approach)
    test_data_source_path = current_wd / "data" / "eeg" / config.target
    test_data_target_path = current_wd / "data" / "datasets" / config.target

    # ==========================================================================
    # 1. Load All Target Subject Data
    # ==========================================================================
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

    # ==========================================================================
    # 2. Create Training Set (Baseline approach: use all 23 files for CV)
    # ==========================================================================
    # BASELINE APPROACH: Use all 23 files for cross-validation only
    # No separate test set - all data is used for CV (test set is "hidden")
    # This matches the baseline: all available data → CV → final model on all data
    x_raw_train, events_train, train_filenames, sub_ids_train, train_indices = create_subject_train_set(
        config,
        all_target_raws,
        all_target_events,
        target_metadata["filenames"],
        num_general=6,
        num_dino=17,
        num_supression=0,
        shuffle=True
    )
    print(f"Created training set with {len(x_raw_train)} files (all files used for CV only, matching baseline approach).")

    # ==========================================================================
    # 3. Preprocess Training Data: Filter, Epoch, Add Metadata
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PREPROCESSING TRAINING DATA")
    print("=" * 60)

    all_epochs_list = []
    for raw, events, sub_id, filename in zip(x_raw_train, events_train, sub_ids_train, train_filenames):
        # FILTERING: Filter the data by using the mne apply function method
        filtered_raw = raw.copy()
        filtered_raw.apply_function(filter_obj.apply_filter_offline)

        # CHANNEL REMOVAL: Remove unnecessary channels like noise sources
        filtered_raw.drop_channels(config.remove_channels)

        # EPOCHING: Create epochs with metadata for CV (consistent with baseline)
        epochs = mne.Epochs(
            filtered_raw,
            events,
            event_id=target_event_id,
            tmin=0.3,  # Start at 0.3s to avoid VEP/ERP from visual cues
            tmax=3.0,  # Match baseline epoching window
            preload=True,
            baseline=None,
        )

        # Skip files with no epochs (avoids "zero-size array" in concatenate_epochs)
        if len(epochs) == 0:
            print(f"  Skipping {filename}: no epochs after epoching.")
            continue

        # Session ID for CV: same session = same fold (extract sub-XXX_ses-YYY from filename)
        session_id = _session_id_from_filename(filename)

        # Attach metadata for grouped CV (by session, matching baseline)
        metadata = pd.DataFrame({
            "subject_id": [sub_id] * len(epochs),
            "session": [session_id] * len(epochs),  # Group by session (not by file)
            "filename": [filename] * len(epochs),
            "condition": epochs.events[:, 2],
        })
        epochs.metadata = metadata
        all_epochs_list.append(epochs)

    # Combine all epochs (require at least one non-empty Epochs object)
    if not all_epochs_list:
        raise ValueError(
            "No epochs after preprocessing. All files had zero epochs (check events/event_id and epoching window)."
        )
    combined_epochs = mne.concatenate_epochs(all_epochs_list)
    
    X_train = combined_epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y_train = combined_epochs.events[:, 2]  # Labels (0, 1, 2) from target_event_id

    # Groups for CV: by session (multiple files can share one session, matching baseline)
    groups = combined_epochs.metadata["session"].values

    # Get unique sessions
    unique_sessions = np.unique(groups)
    n_folds_actual = min(n_folds_target, len(unique_sessions)) if len(unique_sessions) > 0 else n_folds_target

    if n_folds_actual < n_folds_target:
        print(f"WARNING: Only {len(unique_sessions)} sessions available, using {n_folds_actual}-fold CV instead of {n_folds_target}-fold")
    else:
        print(f"Using {n_folds_target}-fold cross-validation (forced)")

    print(f"Training data shape: {X_train.shape}")
    print(f"Labels distribution: {np.unique(y_train, return_counts=True)}")
    print(f"Number of sessions (CV groups): {len(unique_sessions)}")
    print(f"Sessions: {list(unique_sessions)}")
    print(f"Adjusted folds for CV: {n_folds_actual}")

    # Recreate GroupKFold with the correct number of splits (must be <= number of groups)
    if n_folds_actual >= 2:
        gkf = GroupKFold(n_splits=n_folds_actual)
    else:
        gkf = None

    # ==========================================================================
    # 5. Cross-Validation with CombinedAdaptiveLDA (matching baseline methodology)
    # ==========================================================================
    cv_metrics_list = []  # Initialize outside if block for summary table
    
    if n_folds_actual >= 2 and gkf is not None and groups is not None:
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION WITH COMBINED ADAPTIVE LDA")
        print("=" * 60)

        cv_confusion_matrices = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
            print(f"\n--- Fold {fold_idx + 1}/{n_folds_actual} ---")

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
            y_train_fold = fold_windowed["y_train"]  # Labels already [0, 1, 2]
            X_val_fold = fold_windowed["X_val"]
            y_val_fold = fold_windowed["y_val"]  # Labels already [0, 1, 2]

            # Artifact removal within each fold (matching baseline)
            ar = ArtefactRemoval()
            ar.get_rejection_thresholds(X_train_fold, config)

            X_train_clean, y_train_clean = ar.reject_bad_epochs(
                X_train_fold, y_train_fold
            )
            X_val_clean, y_val_clean = ar.reject_bad_epochs(X_val_fold, y_val_fold)

            # Skip if no data left after artifact rejection
            if len(X_train_clean) == 0 or len(X_val_clean) == 0:
                print("Skipped (no data after AR)")
                continue

            # Extract features
            train_features = extract_features(X_train_clean, config.fs)
            val_features = extract_features(X_val_clean, config.fs)

            # Train CombinedAdaptiveLDA with optimized hyperparameters
            fold_clf = CombinedAdaptiveLDA(
                confidence_threshold=0.7,
                ensemble_weight=0.5,
                move_threshold=0.6,  # Optimized threshold
                reg=1e-3,  # Optimized regularization
                shrinkage_alpha=0.05,  # Optimized shrinkage
                uc_mu=0.4 * 2**-7,  # Lower update rate for stability
                use_adaptive_lr=True,
                use_improved_composition=True
            )

            start_train = time.time() * 1000
            fold_clf.fit(train_features, y_train_clean)  # Use cleaned labels after artifact removal
            end_train = time.time() * 1000

            # Predict on validation fold
            start_eval = time.time() * 1000
            fold_preds = fold_clf.predict(val_features)
            end_eval = time.time() * 1000
            fold_probs = fold_clf.predict_proba(val_features)

            # Compute metrics
            fold_metrics = compile_metrics(
                y_true=y_val_clean,
                y_pred=fold_preds,
                y_prob=fold_probs,
                timings={
                    "train_time": end_train - start_train,
                    "infer_latency": (end_eval - start_eval) / max(1, len(y_val_clean)),
                    "total_latency": (end_eval - start_eval) / max(1, len(y_val_clean)),
                    "filter_latency": filter_obj.get_filter_latency(),
                },
                n_classes=3,
            )

            cv_metrics_list.append(fold_metrics)
            cv_confusion_matrices.append(confusion_matrix(y_val_clean, fold_preds))
            print(f"Fold {fold_idx + 1} Accuracy: {fold_metrics['Acc.']:.4f}")

        # CV Summary (matching baseline format)
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS (Mean +/- Std)")
        print("=" * 60)

        # Format CV results like baseline: mean +/- std for key metrics
        cv_result = {"Model": "CombinedAdaptiveLDA"}
        metric_keys = ["Acc.", "B. Acc.", "F1 Score", "ECE", "Brier"]

        for key in metric_keys:
            values = [m[key] for m in cv_metrics_list if key in m]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv_result[key] = f"{mean_val:.4f} +/- {std_val:.4f}"
            else:
                cv_result[key] = "N/A"

        # Add timing info if available
        if cv_metrics_list and "train_time" in cv_metrics_list[0]:
            fold_times = [m.get("train_time", 0) / 1000.0 for m in cv_metrics_list]  # Convert ms to seconds
            cv_result["Avg. Fold Time (s)"] = f"{np.mean(fold_times):.1f}"
        else:
            cv_result["Avg. Fold Time (s)"] = "N/A"

        # Display using MetricsTable (matching baseline)
        cv_table = MetricsTable()
        cv_table.add_rows([cv_result])
        cv_table.display()

        # Save average confusion matrix
        if len(cv_confusion_matrices) > 0:
            avg_cm = np.mean(cv_confusion_matrices, axis=0).astype(int)
            plt.figure(figsize=(8, 6))
            sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Rest', 'Left', 'Right'],
                        yticklabels=['Rest', 'Left', 'Right'])
            plt.xlabel('Predicted', fontsize=12, fontweight='bold')
            plt.ylabel('True', fontsize=12, fontweight='bold')
            plt.title('CombinedAdaptiveLDA CV - Average Confusion Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            cm_path = current_wd / "combined_adaptive_lda_cv_confusion_matrix.png"
            plt.savefig(cm_path, dpi=150)
            print(f"\nCV confusion matrix saved: {cm_path}")
            plt.close()

    # ==========================================================================
    # 6. Train Final CombinedAdaptiveLDA on All Training Data
    # ==========================================================================
    print("\n" + "=" * 60)
    print("TRAINING FINAL COMBINED ADAPTIVE LDA MODEL")
    print("=" * 60)

    # Window all training data
    X_train_windows, y_train_windows, _ = epochs_to_windows(
        combined_epochs,
        groups,
        window_size=config.window_size,
        step_size=config.step_size,
    )
    # Labels are already [0, 1, 2] from epochs_to_windows, no mapping needed

    # Artifact removal (same as in CV, matching baseline)
    ar = ArtefactRemoval()
    ar.get_rejection_thresholds(X_train_windows, config)
    X_train_clean, y_train_clean = ar.reject_bad_epochs(
        X_train_windows, y_train_windows
    )

    if len(X_train_clean) == 0:
        print("Warning: No data left after artifact rejection. Skipping final model training.")
        sys.exit(1)

    # Extract features
    train_features_all = extract_features(X_train_clean, config.fs)

    # Train final model with optimized hyperparameters
    clf = CombinedAdaptiveLDA(
        confidence_threshold=0.7,
        ensemble_weight=0.5,
        move_threshold=0.6,  # Optimized threshold
        reg=1e-3,  # Optimized regularization
        shrinkage_alpha=0.05,  # Optimized shrinkage
        uc_mu=0.4 * 2**-7,  # Lower update rate for stability
        use_adaptive_lr=True,
        use_improved_composition=True
    )

    start_train = time.time() * 1000
    clf.fit(train_features_all, y_train_clean)
    end_train = time.time() * 1000

    print(f"Training completed in {end_train - start_train:.2f} ms")
    print(f"Model stats: {clf.get_stats()}")
    print(f"Stage info: {clf.get_stage_info()}")

    # ==========================================================================
    # 7. Save the Trained Model
    # ==========================================================================
    model_path = current_wd / "resources" / "models" / "combined_adaptive_lda.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump({
            'model': clf,
            'sfreq': config.fs,
            'n_features': clf.n_features_,
            'stage_info': clf.get_stage_info(),
            'stats': clf.get_stats(),
        }, f)
    print(f"\nCombined Adaptive LDA model saved to: {model_path}")

    # Final summary (matching baseline)
    print("\n" + "=" * 80)
    print("FINAL METRICS SUMMARY")
    print("=" * 80)
    print(f"Cross-Validation: {n_folds_actual}-fold (session-wise grouping)")
    print(f"Random State: {config.random_state} (for reproducibility)")
    if 'cv_metrics_list' in locals() and len(cv_metrics_list) > 0:
        cv_mean_metrics = {}
        for key in cv_metrics_list[0].keys():
            values = [m[key] for m in cv_metrics_list]
            cv_mean_metrics[key] = np.mean(values)
        print(f"Mean CV Accuracy: {cv_mean_metrics.get('Acc.', 0):.4f}")
        print(f"Mean CV F1 Score: {cv_mean_metrics.get('F1 Score', 0):.4f}")
    print("=" * 80)

    print("\n" + "=" * 60)
    print("COMBINED ADAPTIVE LDA OFFLINE EVALUATION COMPLETE!")
    print("=" * 60)

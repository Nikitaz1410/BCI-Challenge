"""
Offline Training and Testing Pipeline for the Baseline (AllRounder) Model.

This script compares multiple feature/classifier configurations using
session-wise grouped cross-validation.

Pipeline:
1. Load configuration from YAML
2. Load target subject data
3. Filter and preprocess the EEG data
4. Create epochs with metadata for grouped CV (by session)
5. For each model configuration:
   - Perform K-Fold cross-validation
   - Collect metrics
6. Compare all configurations in a summary table

Supported feature extractors:
- CSP (Common Spatial Patterns)
- Welch band power
- Lateralization features

Supported classifiers:
- LDA (Linear Discriminant Analysis)
- SVM (Support Vector Machine)
- Logistic Regression
- Random Forest

Usage:
    python main_offline_Baseline.py
"""

import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add src directory to Python path to allow imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Evaluation
from bci.Evaluation.metrics import MetricsTable, compile_metrics

# Data Acquisition
from bci.loading.loading import (
    create_subject_test_set,
    create_subject_train_set,
    load_target_subject_data,
)

# Preprocessing
from bci.Preprocessing.artefact_removal import ArtefactRemoval
from bci.Preprocessing.filters import Filter
from bci.Preprocessing.windows import epochs_to_windows, epochs_windows_from_fold

# Models - HybridLDA wrapper for testing different features
from bci.models.AdaptiveLDA_modules.hybrid_lda_wrapper import HybridLDAWrapper

# Utils
from bci.utils.bci_config import load_config
from bci.utils.utils import choose_model


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
# Model Configurations to Compare
# =============================================================================
MODEL_CONFIGURATIONS: List[Dict[str, Any]] = [
    # CSP + LDA 
    {
        "name": "CSP + LDA",
        "features": "csp",
        "classifier": "lda",
        "scale": True,
    },
    # CSP + SVM
    {
        "name": "CSP + SVM",
        "features": "csp",
        "classifier": "svm",
        "scale": True,
    },
    # CSP + Logistic Regression
    {
        "name": "CSP + LogReg",
        "features": "csp",
        "classifier": "logreg",
        "scale": True,
    },
    # Band power + LDA
    {
        "name": "BandPower + LDA",
        "features": "welch_bandpower",
        "classifier": "lda",
        "scale": True,
    },
    # Band power + SVM
    {
        "name": "BandPower + SVM",
        "features": "welch_bandpower",
        "classifier": "svm",
        "scale": True,
    },
    # CSP + Band power + LDA (combined features)
    {
        "name": "CSP+BP + LDA",
        "features": ["csp", "welch_bandpower"],
        "classifier": "lda",
        "scale": True,
    },
    # CSP + Band power + SVM
    {
        "name": "CSP+BP + SVM",
        "features": ["csp", "welch_bandpower"],
        "classifier": "svm",
        "scale": True,
    },
    # CSP + Band power + Random Forest
    {
        "name": "CSP+BP + RF",
        "features": ["csp", "welch_bandpower"],
        "classifier": "rf",
        "scale": True,
    },
    # HybridLDA with different feature types
    {
        "name": "HybridLDA + Log-Bandpower",
        "features": "log_bandpower",
        "classifier": "hybrid_lda",
        "scale": False,
    },
    {
        "name": "HybridLDA + CSP",
        "features": "csp",
        "classifier": "hybrid_lda",
        "scale": False,
    },
    {
        "name": "HybridLDA + Welch-Bandpower",
        "features": "welch_bandpower",
        "classifier": "hybrid_lda",
        "scale": False,
    },
    {
        "name": "HybridLDA + CSP+BP",
        "features": ["csp", "welch_bandpower"],
        "classifier": "hybrid_lda",
        "scale": False,
    },
]


def run_cv_for_config(
    model_config: Dict[str, Any],
    combined_epochs: mne.Epochs,
    groups: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Any,
    n_classes: int,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """
    Run cross-validation for a single model configuration.

    Parameters
    ----------
    model_config : dict
        Model configuration with 'name', 'features', 'classifier', 'scale'
    combined_epochs : mne.Epochs
        Combined epochs for all training data
    groups : np.ndarray
        Group labels for CV (e.g., session/file names)
    X_train : np.ndarray
        Training data (n_epochs, n_channels, n_times)
    y_train : np.ndarray
        Training labels
    config : EEGConfig
        Configuration object with window_size, step_size, etc.
    n_classes : int
        Number of classes
    n_folds : int
        Number of CV folds

    Returns
    -------
    dict
        Dictionary with model name and mean/std metrics
    """
    config_name = model_config["name"]
    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}")
    print(f"{'='*60}")

    gkf = GroupKFold(n_splits=n_folds)
    cv_metrics_list = []
    fold_times = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X_train, y_train, groups=groups)
    ):
        fold_start = time.time()
        print(f"  Fold {fold_idx + 1}/{n_folds}...", end=" ")

        # Extract windowed epochs for this fold
        fold_windowed_epochs = epochs_windows_from_fold(
            combined_epochs,
            groups,
            train_idx,
            val_idx,
            window_size=config.window_size,
            step_size=config.step_size,
        )

        X_train_fold = fold_windowed_epochs["X_train"]
        y_train_fold = fold_windowed_epochs["y_train"]
        X_val_fold = fold_windowed_epochs["X_val"]
        y_val_fold = fold_windowed_epochs["y_val"]

        # Artifact removal within each fold
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

        try:
            # Handle HybridLDA differently (uses wrapper with feature extraction)
            if model_config["classifier"] == "hybrid_lda":
                clf = HybridLDAWrapper(
                    features=model_config["features"],
                    move_threshold=0.5,
                    reg=1e-2,
                    shrinkage_alpha=0.1,
                    uc_mu=0.4 * 2**-6,
                    sfreq=config.fs  # Pass sampling frequency
                )
                clf.fit(X_train_clean, y_train_clean)
                
                # Evaluate on validation fold
                fold_predictions = clf.predict(X_val_clean)
                fold_probabilities = clf.predict_proba(X_val_clean)
            else:
                # Baseline models path (existing code)
                clf = choose_model(
                    "baseline",
                    {
                        "features": model_config["features"],
                        "classifier": model_config["classifier"],
                        "scale": model_config["scale"],
                        "random_state": config.random_state,
                    },
                )
                
                clf.fit(X_train_clean, y_train_clean)
                
                # Evaluate on validation fold
                fold_predictions = clf.predict(X_val_clean)
                fold_probabilities = clf.predict_proba(X_val_clean)

            # Compute metrics
            fold_metrics = compile_metrics(
                y_true=y_val_clean,
                y_pred=fold_predictions,
                y_prob=fold_probabilities,
                timings=None,
                n_classes=n_classes,
            )

            cv_metrics_list.append(fold_metrics)
            fold_time = time.time() - fold_start
            fold_times.append(fold_time)
            print(f"Acc: {fold_metrics['Acc.']:.4f} ({fold_time:.1f}s)")

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Aggregate results
    if len(cv_metrics_list) == 0:
        return {
            "Model": config_name,
            "Acc.": "N/A",
            "B. Acc.": "N/A",
            "F1 Score": "N/A",
            "ECE": "N/A",
            "Brier": "N/A",
        }

    # Compute mean and std for each metric
    result = {"Model": config_name}
    metric_keys = ["Acc.", "B. Acc.", "F1 Score", "ECE", "Brier"]

    for key in metric_keys:
        values = [m[key] for m in cv_metrics_list if key in m]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            result[key] = f"{mean_val:.4f} +/- {std_val:.4f}"
        else:
            result[key] = "N/A"

    # Add timing info
    result["Avg. Fold Time (s)"] = f"{np.mean(fold_times):.1f}"

    return result


def run_baseline_comparison_pipeline():
    """
    Main pipeline that compares multiple model configurations using
    session-wise grouped cross-validation.
    """
    # =========================================================================
    # 1. Load Configuration
    # =========================================================================
    # Detect project root: handle both workspace root and BCI-Challenge subdirectory
    # Script is at: [workspace]/BCI-Challenge/src/bci/main_offline_Baseline.py
    script_dir = Path(__file__).parent.parent.parent  # Goes up 3 levels from script
    
    # Check if we're in a BCI-Challenge subdirectory (workspace structure)
    # The script is at: BCI-Challenge/src/bci/main_offline_Baseline.py
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

    # Number of folds = number of sessions (leave-one-session-out CV)
    print("Using session-wise cross-validation (n_folds = n_sessions).")

    # Initialize filter
    filter_obj = Filter(config, online=False)

    # Setup paths
    test_data_source_path = current_wd / "data" / "eeg" / config.target
    test_data_target_path = current_wd / "data" / "datasets" / config.target

    # =========================================================================
    # 2. Load Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Load all target subject data
    (
        all_target_raws,
        all_target_events,
        target_event_id,
        target_sub_ids,
        target_metadata,
    ) = load_target_subject_data(
        root=current_wd,
        source_path=test_data_source_path,
        target_path=test_data_target_path,
        resample=None,
    )

    print(f"Loaded {len(all_target_raws)} sessions from target subject data.")

    # Create training set from target subject data (use all available data for CV)
    x_raw_train, events_train, train_filenames, sub_ids_train, train_indices = (
        create_subject_train_set(
            config,
            all_target_raws,
            all_target_events,
            target_metadata["filenames"],
            num_general=3,
            num_dino=17,
            num_supression=0,
            shuffle=True,
        )
    )
    print(f"Using {len(x_raw_train)} sessions for cross-validation.")

    # =========================================================================
    # 3. Preprocessing: Filter and Create Epochs
    # =========================================================================
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    all_epochs_list = []
    for raw, events, sub_id, filename in zip(
        x_raw_train, events_train, sub_ids_train, train_filenames
    ):
        # FILTERING: Apply bandpass filter
        filtered_raw = raw.copy()
        filtered_raw.apply_function(filter_obj.apply_filter_offline)

        # CHANNEL REMOVAL: Remove unnecessary channels (noise sources)
        filtered_raw.drop_channels(config.remove_channels)

        # EPOCHING: Create epochs with metadata for CV
        epochs = mne.Epochs(
            filtered_raw,
            events,
            event_id=target_event_id,
            tmin=0.3,  # Start at 0.3s to avoid VEP/ERP from visual cues
            tmax=3.0,
            preload=True,
            baseline=None,
        )

        # Skip files with no epochs (avoids "zero-size array" in concatenate_epochs)
        if len(epochs) == 0:
            print(f"  Skipping {filename}: no epochs after epoching.")
            continue

        # Session ID for CV: same session = same fold (extract sub-XXX_ses-YYY from filename)
        session_id = _session_id_from_filename(filename)

        # Attach metadata for grouped CV (by session)
        metadata = pd.DataFrame(
            {
                "subject_id": [sub_id] * len(epochs),
                "session": [session_id] * len(epochs),  # Group by session (not by file)
                "filename": [filename] * len(epochs),
                "condition": epochs.events[:, 2],
            }
        )
        epochs.metadata = metadata
        all_epochs_list.append(epochs)

    # Combine all epochs (require at least one non-empty Epochs object)
    if not all_epochs_list:
        raise ValueError(
            "No epochs after preprocessing. All files had zero epochs (check events/event_id and epoching window)."
        )
    combined_epochs = mne.concatenate_epochs(all_epochs_list)
    X_train = combined_epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y_train = combined_epochs.events[:, 2]  # Labels (e.g., 0, 1, 2)

    # Groups for CV: by session (multiple files can share one session)
    groups = combined_epochs.metadata["session"].values

    # Get unique sessions
    unique_sessions = np.unique(groups)
    n_folds = len(unique_sessions)  # One fold per session (leave-one-session-out)

    print(f"Training data shape: {X_train.shape}")
    print(f"Labels distribution: {np.unique(y_train, return_counts=True)}")
    print(f"Number of sessions (CV groups): {len(unique_sessions)}")
    print(f"Sessions: {list(unique_sessions)}")
    print(f"Adjusted folds for CV: {n_folds}")

    # =========================================================================
    # 4. Run Cross-Validation for Each Configuration
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPARING MODEL CONFIGURATIONS")
    print("=" * 60)
    print(f"Total configurations to evaluate: {len(MODEL_CONFIGURATIONS)}")

    all_results = []
    n_classes = len(target_event_id)

    for config_idx, model_config in enumerate(MODEL_CONFIGURATIONS):
        print(f"\n[{config_idx + 1}/{len(MODEL_CONFIGURATIONS)}] ", end="")

        result = run_cv_for_config(
            model_config=model_config,
            combined_epochs=combined_epochs,
            groups=groups,
            X_train=X_train,
            y_train=y_train,
            config=config,
            n_classes=n_classes,
            n_folds=n_folds,
        )

        all_results.append(result)

    # =========================================================================
    # 5. Sort by F1 Score and Display Comparison Table
    # =========================================================================
    def _f1_sort_key(result: Dict[str, Any]) -> float:
        """Extract mean F1 score for sorting; use -1 for N/A so they sort last."""
        f1_str = result.get("F1 Score", "N/A")
        if f1_str == "N/A":
            return -1.0
        try:
            return float(f1_str.split("+/-")[0].strip())
        except (ValueError, IndexError):
            return -1.0

    all_results_sorted = sorted(
        all_results, key=_f1_sort_key, reverse=True
    )

    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS (sorted by F1 Score, descending)")
    print("=" * 80)
    print(f"Cross-Validation: {n_folds}-fold, grouped by session (sub-XXX_ses-YYY)")
    print(f"Total sessions: {len(unique_sessions)}")
    print(f"Total epochs: {len(X_train)}")
    print("=" * 80)

    # Create comparison table (sorted by F1 score)
    comparison_table = MetricsTable()
    comparison_table.add_rows(all_results_sorted)
    comparison_table.display()

    # Also create a pandas DataFrame for easier analysis (sorted)
    df_results = pd.DataFrame(all_results_sorted)

    # =========================================================================
    # 6. Find Best Configuration and Train Final Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION (by F1 Score)")
    print("=" * 60)

    # Best is first in sorted list (by F1 score)
    best_result = all_results_sorted[0] if all_results_sorted else None
    best_f1 = _f1_sort_key(best_result) if best_result else -1.0
    best_config = None
    best_config_name = best_result.get("Model", "") if best_result else ""

    if best_result and best_f1 >= 0:
        for model_config in MODEL_CONFIGURATIONS:
            if model_config["name"] == best_config_name:
                best_config = model_config
                break

    if best_config:
        print(f"Best model: {best_config_name}")
        print(f"Mean CV F1 Score: {best_f1:.4f}")
        print(f"Configuration: {best_config}")

        # Train final model with best configuration on all data
        print("\nTraining final model with best configuration...")

        X_train_windows, y_train_windows, _ = epochs_to_windows(
            combined_epochs,
            groups,
            window_size=config.window_size,
            step_size=config.step_size,
        )

        # Save the best model
        model_dir = current_wd / "resources" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        if best_config["classifier"] == "hybrid_lda":
            # Use HybridLDAWrapper with feature extraction
            # Note: use_improved_composition defaults to True in HybridLDAWrapper
            final_clf = HybridLDAWrapper(
                features=best_config["features"],
                move_threshold=0.5,
                reg=1e-2,
                shrinkage_alpha=0.1,
                uc_mu=0.4 * 2**-6,
                sfreq=config.fs
            )
            final_clf.fit(X_train_windows, y_train_windows)
            
            # Save HybridLDA model
            model_path = model_dir / "hybrid_lda_best.pkl"
            final_clf.save(str(model_path))
            print(f"Best HybridLDA model saved to: {model_path}")
        else:
            # Baseline models path (existing code)
            # Artifact removal (same as in CV)
            ar = ArtefactRemoval()
            ar.get_rejection_thresholds(X_train_windows, config)
            X_train_clean, y_train_clean = ar.reject_bad_epochs(
                X_train_windows, y_train_windows
            )

            if len(X_train_clean) == 0:
                print("Warning: No data left after artifact rejection. Skipping final model save.")
            else:
                final_clf = choose_model(
                    "baseline",
                    {
                        "features": best_config["features"],
                        "classifier": best_config["classifier"],
                        "scale": best_config["scale"],
                        "random_state": config.random_state,
                    },
                )

                final_clf.fit(X_train_clean, y_train_clean)

                # Save the best model and ArtefactRemoval
                model_path = model_dir / "baseline_best_model.pkl"
                artefact_path = model_dir / "artefact_removal.pkl"

                final_clf.save(str(model_path))
                with open(artefact_path, "wb") as f:
                    pickle.dump(ar, f)

                print(f"Best model saved to: {model_path}")
                print(f"ArtefactRemoval saved to: {artefact_path}")

    return all_results, df_results


if __name__ == "__main__":
    all_results, df_results = run_baseline_comparison_pipeline()

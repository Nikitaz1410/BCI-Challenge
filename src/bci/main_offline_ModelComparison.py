"""
Model Comparison Script - Evaluates all BCI models for competition selection.

This script runs cross-validation on multiple models:
1. HybridLDA (2-stage adaptive LDA)
2. Combined Adaptive LDA (Hybrid + Core LDA)
3. Riemannian Classifier

Uses the same evaluation methodology as friends' baseline:
- 11-fold cross-validation (session-wise grouping)
- Artifact removal
- Same data splits
- Comprehensive metrics (Accuracy, Balanced Acc, F1, ECE, Brier)

Creates a summary table at the end to help select the best model for competition.
"""

import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Evaluation
from bci.evaluation.metrics import MetricsTable, compile_metrics

# Data Acquisition
from bci.loading.loading import (
    create_subject_train_set,
    load_target_subject_data,
)

# Preprocessing
from bci.preprocessing.artefact_removal import ArtefactRemoval
from bci.preprocessing.filters import Filter
from bci.preprocessing.windows import epochs_to_windows, epochs_windows_from_fold

# Models
from bci.models.riemann import RiemannianClf
from bci.models.adaptive_lda_modules.hybrid_lda_wrapper import HybridLDAWrapper
from bci.models.adaptive_lda_modules.combined_adaptive_lda_wrapper import CombinedAdaptiveLDAWrapper
from bci.models.adaptive_lda_modules.feature_extraction import extract_log_bandpower_features

# Utils
from bci.utils.bci_config import load_config


def _session_id_from_filename(filename: str) -> str:
    """Extract session identifier from BIDS-style filename for CV grouping."""
    base = filename.replace("_raw", "").strip("_")
    match = re.match(r"(sub-[^_]+_ses-[^_]+)", base)
    if match:
        return match.group(1)
    return filename


def extract_features(signals, sfreq):
    """Extract log-bandpower features."""
    return extract_log_bandpower_features(signals, sfreq=sfreq, mu_band=(8, 12), beta_band=(13, 30))


def run_cv_for_model(
    model_name: str,
    model_config: Dict[str, Any],
    combined_epochs: mne.Epochs,
    groups: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Any,
    n_classes: int,
    n_folds: int,
) -> Dict[str, Any]:
    """Run cross-validation for a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
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
            # Instantiate and train model based on type
            if model_config["type"] == "hybrid_lda":
                clf = HybridLDAWrapper(
                    features=model_config.get("features", "log_bandpower"),
                    move_threshold=model_config.get("move_threshold", 0.6),
                    reg=model_config.get("reg", 1e-3),
                    shrinkage_alpha=model_config.get("shrinkage_alpha", 0.05),
                    uc_mu=model_config.get("uc_mu", 0.4 * 2**-6),
                    sfreq=config.fs,
                    use_improved_composition=True
                )
                clf.fit(X_train_clean, y_train_clean)
                fold_predictions = clf.predict(X_val_clean)
                fold_probabilities = clf.predict_proba(X_val_clean)
                
            elif model_config["type"] == "combined_adaptive_lda":
                clf = CombinedAdaptiveLDAWrapper(
                    features=model_config.get("features", "log_bandpower"),
                    confidence_threshold=model_config.get("confidence_threshold", 0.7),
                    ensemble_weight=model_config.get("ensemble_weight", 0.5),
                    move_threshold=model_config.get("move_threshold", 0.6),
                    reg=model_config.get("reg", 1e-3),
                    shrinkage_alpha=model_config.get("shrinkage_alpha", 0.05),
                    uc_mu=model_config.get("uc_mu", 0.4 * 2**-7),
                    use_adaptive_lr=True,
                    sfreq=config.fs,
                    use_improved_composition=True
                )
                clf.fit(X_train_clean, y_train_clean)
                fold_predictions = clf.predict(X_val_clean)
                fold_probabilities = clf.predict_proba(X_val_clean)
                
            elif model_config["type"] == "riemann":
                clf = RiemannianClf(cov_est="lwf")
                clf.fit(X_train_clean, y_train_clean)
                # RiemannianClf.predict returns (predictions, cov), so extract just predictions
                fold_predictions, _ = clf.predict(X_val_clean)
                # RiemannianClf.predict_proba returns (probabilities, cov), so extract just probabilities
                fold_probabilities, _ = clf.predict_proba(X_val_clean)
                
            else:
                raise ValueError(f"Unknown model type: {model_config['type']}")

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
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results
    if len(cv_metrics_list) == 0:
        return {
            "Model": model_name,
            "Acc.": "N/A",
            "B. Acc.": "N/A",
            "F1 Score": "N/A",
            "ECE": "N/A",
            "Brier": "N/A",
        }

    # Compute mean and std for each metric
    result = {"Model": model_name}
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


def main():
    """Main comparison pipeline."""
    # =========================================================================
    # 1. Load Configuration
    # =========================================================================
    script_dir = Path(__file__).parent.parent.parent
    
    if (script_dir / "src").exists() and (script_dir / "data").exists():
        current_wd = script_dir
    elif (script_dir / "BCI-Challenge" / "src").exists() and (script_dir / "BCI-Challenge" / "data").exists():
        current_wd = script_dir / "BCI-Challenge"
    else:
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
    # Note: GroupKFold doesn't use random_state, but grouping is deterministic based on session IDs
    
    # Force 11-fold CV
    n_folds_target = 11
    print(f"Using {n_folds_target}-fold cross-validation (forced)")
    print(f"Random state: {config.random_state} (for reproducibility)")

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

    all_target_raws, all_target_events, target_event_id, target_sub_ids, target_metadata = (
        load_target_subject_data(
            root=current_wd,
            source_path=test_data_source_path,
            target_path=test_data_target_path,
            resample=None,
        )
    )

    print(f"Loaded {len(all_target_raws)} sessions from target subject data.")

    # Create training set (use all 23 files for CV)
    x_raw_train, events_train, train_filenames, sub_ids_train, train_indices = (
        create_subject_train_set(
            config,
            all_target_raws,
            all_target_events,
            target_metadata["filenames"],
            num_general=6,
            num_dino=17,
            num_supression=0,
            shuffle=True,
        )
    )
    print(f"Using {len(x_raw_train)} sessions for cross-validation.")

    # =========================================================================
    # 3. Preprocessing
    # =========================================================================
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    all_epochs_list = []
    for raw, events, sub_id, filename in zip(
        x_raw_train, events_train, sub_ids_train, train_filenames
    ):
        filtered_raw = raw.copy()
        filtered_raw.apply_function(filter_obj.apply_filter_offline)
        filtered_raw.drop_channels(config.remove_channels)

        epochs = mne.Epochs(
            filtered_raw,
            events,
            event_id=target_event_id,
            tmin=0.3,
            tmax=3.0,
            preload=True,
            baseline=None,
        )

        if len(epochs) == 0:
            print(f"  Skipping {filename}: no epochs after epoching.")
            continue

        session_id = _session_id_from_filename(filename)
        metadata = pd.DataFrame(
            {
                "subject_id": [sub_id] * len(epochs),
                "session": [session_id] * len(epochs),
                "filename": [filename] * len(epochs),
                "condition": epochs.events[:, 2],
            }
        )
        epochs.metadata = metadata
        all_epochs_list.append(epochs)

    if not all_epochs_list:
        raise ValueError("No epochs after preprocessing.")

    combined_epochs = mne.concatenate_epochs(all_epochs_list)
    X_train = combined_epochs.get_data()
    y_train = combined_epochs.events[:, 2]
    groups = combined_epochs.metadata["session"].values

    unique_sessions = np.unique(groups)
    n_sessions = len(unique_sessions)
    n_folds = min(n_folds_target, n_sessions) if n_sessions > 0 else n_folds_target

    if n_folds < n_folds_target:
        print(f"WARNING: Only {n_sessions} sessions available, using {n_folds}-fold CV")
    else:
        print(f"Using {n_folds_target}-fold cross-validation")

    print(f"Training data shape: {X_train.shape}")
    print(f"Labels distribution: {np.unique(y_train, return_counts=True)}")
    print(f"Number of sessions: {n_sessions}")

    # =========================================================================
    # 4. Define Models to Compare
    # =========================================================================
    MODEL_CONFIGURATIONS = [
        # Adaptive LDA models
        {"name": "HybridLDA", "type": "hybrid_lda", "features": "log_bandpower"},
        {"name": "CombinedAdaptiveLDA", "type": "combined_adaptive_lda", "features": "log_bandpower"},
        
        # Riemannian
        {"name": "Riemannian", "type": "riemann"},
    ]

    # =========================================================================
    # 5. Run Cross-Validation for Each Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - CROSS-VALIDATION")
    print("=" * 80)
    print(f"Total models to evaluate: {len(MODEL_CONFIGURATIONS)}")

    all_results = []
    n_classes = len(target_event_id)

    for model_config in MODEL_CONFIGURATIONS:
        result = run_cv_for_model(
            model_name=model_config["name"],
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
    # 6. Sort by F1 Score and Display Results
    # =========================================================================
    def _f1_sort_key(result: Dict[str, Any]) -> float:
        """Extract mean F1 score for sorting."""
        f1_str = result.get("F1 Score", "N/A")
        if f1_str == "N/A":
            return -1.0
        try:
            return float(f1_str.split("+/-")[0].strip())
        except (ValueError, IndexError):
            return -1.0

    all_results_sorted = sorted(all_results, key=_f1_sort_key, reverse=True)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS (sorted by F1 Score, descending)")
    print("=" * 80)
    print(f"Cross-Validation: {n_folds}-fold, grouped by session")
    print(f"Total sessions: {n_sessions}")
    print(f"Total epochs: {len(X_train)}")
    print("=" * 80)

    # Create comparison table
    comparison_table = MetricsTable()
    comparison_table.add_rows(all_results_sorted)
    comparison_table.display()

    # Save to CSV
    df_results = pd.DataFrame(all_results_sorted)
    summary_path = current_wd / "model_comparison_results.csv"
    df_results.to_csv(summary_path, index=False)
    print(f"\nResults saved to: {summary_path}")

    # Print final metrics summary
    print("\n" + "=" * 80)
    print("FINAL METRICS SUMMARY")
    print("=" * 80)
    print(f"Cross-Validation: {n_folds}-fold (session-wise grouping)")
    print(f"Random State: {config.random_state} (for reproducibility)")
    print(f"Total Models Evaluated: {len(all_results_sorted)}")
    print("=" * 80)
    
    # Print recommendation
    if all_results_sorted:
        best_model = all_results_sorted[0]
        print("\n" + "=" * 80)
        print("RECOMMENDATION FOR COMPETITION")
        print("=" * 80)
        print(f"Best Model (by F1 Score): {best_model['Model']}")
        print(f"  Accuracy:      {best_model.get('Acc.', 'N/A')}")
        print(f"  Balanced Acc.: {best_model.get('B. Acc.', 'N/A')}")
        print(f"  F1 Score:      {best_model.get('F1 Score', 'N/A')}")
        print(f"  ECE:           {best_model.get('ECE', 'N/A')}")
        print(f"  Brier Score:   {best_model.get('Brier', 'N/A')}")
        print("=" * 80)

    return all_results_sorted, df_results


if __name__ == "__main__":
    all_results, df_results = main()

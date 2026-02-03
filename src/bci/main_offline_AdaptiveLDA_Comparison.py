"""
Offline Training and Testing Pipeline for AdaptiveLDA Model Comparison.

This script compares multiple HybridLDA and CombinedAdaptiveLDA configurations using
session-wise grouped cross-validation. Includes best combination comparison.

Pipeline:
1. Load configuration from YAML
2. Load target subject data
3. Filter and preprocess the EEG data
4. Create epochs with metadata for grouped CV (by session)
5. For each model configuration:
   - Perform K-Fold cross-validation
   - Collect metrics
6. Compare all configurations in a summary table
7. Compare best HybridLDA vs best CombinedAdaptiveLDA

Supported feature extractors:
- CSP (Common Spatial Patterns)
- Welch band power
- Log-bandpower
- Combined features

Usage:
    python main_offline_AdaptiveLDA_Comparison.py
"""

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
from bci.evaluation.metrics import MetricsTable, compile_metrics

# Data Acquisition
from bci.loading.loading import (
    create_subject_test_set,
    create_subject_train_set,
    load_target_subject_data,
)

# Preprocessing
from bci.preprocessing.artefact_removal import ArtefactRemoval
from bci.preprocessing.filters import Filter
from bci.preprocessing.windows import epochs_to_windows, epochs_windows_from_fold

# Models - HybridLDA and CombinedAdaptiveLDA wrappers
from bci.models.adaptive_lda_modules.hybrid_lda_wrapper import HybridLDAWrapper
from bci.models.adaptive_lda_modules.combined_adaptive_lda_wrapper import CombinedAdaptiveLDAWrapper

# Utils
from bci.utils.bci_config import load_config


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
# Model Configurations to Compare (HybridLDA and CombinedAdaptiveLDA)
# =============================================================================
MODEL_CONFIGURATIONS: List[Dict[str, Any]] = [
    # HybridLDA Configurations
    {
        "name": "HybridLDA + Log-Bandpower",
        "type": "hybrid_lda",
        "features": "log_bandpower",
        "move_threshold": 0.5,
        "reg": 1e-2,
        "shrinkage_alpha": 0.1,
        "uc_mu": 0.4 * 2**-6,
    },
    {
        "name": "HybridLDA + Log-BP (thresh=0.6)",
        "type": "hybrid_lda",
        "features": "log_bandpower",
        "move_threshold": 0.6,
        "reg": 1e-2,
        "shrinkage_alpha": 0.1,
        "uc_mu": 0.4 * 2**-6,
    },
    {
        "name": "HybridLDA + Log-BP (reg=1e-3)",
        "type": "hybrid_lda",
        "features": "log_bandpower",
        "move_threshold": 0.5,
        "reg": 1e-3,
        "shrinkage_alpha": 0.1,
        "uc_mu": 0.4 * 2**-6,
    },
    # CombinedAdaptiveLDA Configurations (winning model)
    {
        "name": "CombinedAdaptiveLDA + Log-Bandpower (default)",
        "type": "combined_adaptive_lda",
        "features": "log_bandpower",
        "confidence_threshold": 0.7,
        "ensemble_weight": 0.5,
        "move_threshold": 0.6,
        "reg": 1e-3,
        "shrinkage_alpha": 0.05,
        "uc_mu": 0.4 * 2**-7,
        "use_adaptive_lr": True,
    },
    {
        "name": "CombinedAdaptiveLDA + Log-BP (thresh=0.5)",
        "type": "combined_adaptive_lda",
        "features": "log_bandpower",
        "confidence_threshold": 0.7,
        "ensemble_weight": 0.5,
        "move_threshold": 0.5,
        "reg": 1e-3,
        "shrinkage_alpha": 0.05,
        "uc_mu": 0.4 * 2**-7,
        "use_adaptive_lr": True,
    },
    {
        "name": "CombinedAdaptiveLDA + Log-BP (conf=0.8)",
        "type": "combined_adaptive_lda",
        "features": "log_bandpower",
        "confidence_threshold": 0.8,
        "ensemble_weight": 0.5,
        "move_threshold": 0.6,
        "reg": 1e-3,
        "shrinkage_alpha": 0.05,
        "uc_mu": 0.4 * 2**-7,
        "use_adaptive_lr": True,
    },
    {
        "name": "CombinedAdaptiveLDA + Log-BP (ensemble=0.7)",
        "type": "combined_adaptive_lda",
        "features": "log_bandpower",
        "confidence_threshold": 0.7,
        "ensemble_weight": 0.7,
        "move_threshold": 0.6,
        "reg": 1e-3,
        "shrinkage_alpha": 0.05,
        "uc_mu": 0.4 * 2**-7,
        "use_adaptive_lr": True,
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
    Run cross-validation for a single model configuration (HybridLDA or CombinedAdaptiveLDA).

    Parameters
    ----------
    model_config : dict
        Model configuration with 'name', 'type', 'features', 'move_threshold', 'reg', etc.
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

    # Use GroupKFold with forced n_splits for reproducibility
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
            # Create model based on type
            if model_config["type"] == "hybrid_lda":
                clf = HybridLDAWrapper(
                    features=model_config["features"],
                    move_threshold=model_config["move_threshold"],
                    reg=model_config["reg"],
                    shrinkage_alpha=model_config["shrinkage_alpha"],
                    uc_mu=model_config["uc_mu"],
                    sfreq=config.fs,
                    use_improved_composition=True
                )
            elif model_config["type"] == "combined_adaptive_lda":
                clf = CombinedAdaptiveLDAWrapper(
                    features=model_config["features"],
                    confidence_threshold=model_config.get("confidence_threshold", 0.7),
                    ensemble_weight=model_config.get("ensemble_weight", 0.5),
                    move_threshold=model_config["move_threshold"],
                    reg=model_config["reg"],
                    shrinkage_alpha=model_config["shrinkage_alpha"],
                    uc_mu=model_config["uc_mu"],
                    use_adaptive_lr=model_config.get("use_adaptive_lr", True),
                    sfreq=config.fs,
                    use_improved_composition=True
                )
            else:
                raise ValueError(f"Unknown model type: {model_config['type']}")
            
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


def run_hybridlda_comparison_pipeline():
    """
    Main pipeline that compares multiple HybridLDA and CombinedAdaptiveLDA 
    configurations using session-wise grouped cross-validation.
    Includes best combination comparison (best HybridLDA vs best CombinedAdaptiveLDA).
    """
    # =========================================================================
    # 1. Load Configuration
    # =========================================================================
    # Detect project root: handle both workspace root and BCI-Challenge subdirectory
    script_dir = Path(__file__).parent.parent.parent  # Goes up 3 levels from script
    
    # Check if we're in a BCI-Challenge subdirectory (workspace structure)
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
    # Note: GroupKFold doesn't use random_state, but we set it for other sklearn functions
    # MNE also respects numpy random state
    
    # Force 11-fold CV
    n_folds_target = 11
    print(f"Using {n_folds_target}-fold cross-validation (forced).")
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
    # Ensure exactly 19 dino and 6 general files for reproducibility
    x_raw_train, events_train, train_filenames, sub_ids_train, train_indices = (
        create_subject_train_set(
            config,
            all_target_raws,
            all_target_events,
            target_metadata["filenames"],
            num_general=6,
            num_dino=19,
            num_supression=0,
            shuffle=True,
        )
    )
    print(f"Using {len(x_raw_train)} sessions for cross-validation.")
    print(f"  - General files: 6 (requested)")
    print(f"  - Dino files: 19 (requested)")
    
    # Create test set (holdout for final evaluation)
    # Try to get test data, but handle gracefully if not enough files available
    try:
        x_raw_test, events_test, test_filenames, sub_ids_test, _ = (
            create_subject_test_set(
                config,
                all_target_raws,
                all_target_events,
                target_metadata["filenames"],
                exclude_indices=train_indices,
                num_general=0,
                num_dino=4,
                num_supression=0,
                shuffle=False,
            )
        )
        print(f"Using {len(x_raw_test)} sessions for final test evaluation.")
    except ValueError as e:
        # If not enough files available for testing, use empty test set
        print(f"WARNING: Could not create test set: {e}")
        print("  Continuing without test set evaluation.")
        x_raw_test = []
        events_test = []
        test_filenames = []
        sub_ids_test = []

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
    n_sessions = len(unique_sessions)
    
    # Force 11-fold CV
    n_folds = n_folds_target
    if n_sessions < n_folds:
        print(f"WARNING: Only {n_sessions} sessions available, but {n_folds} folds requested.")
        print(f"  Using {n_sessions}-fold CV instead (leave-one-session-out).")
        n_folds = n_sessions
    elif n_sessions > n_folds:
        print(f"INFO: {n_sessions} sessions available, using {n_folds}-fold CV.")
        print(f"  Some sessions will be grouped together in folds.")

    print(f"Training data shape: {X_train.shape}")
    print(f"Labels distribution: {np.unique(y_train, return_counts=True)}")
    print(f"Number of sessions (CV groups): {n_sessions}")
    print(f"Sessions: {list(unique_sessions)}")
    print(f"Using {n_folds}-fold cross-validation")

    # =========================================================================
    # 4. Run Cross-Validation for Each Model Configuration
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
    print("MODEL CONFIGURATION COMPARISON (sorted by F1 Score, descending)")
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
    # 6. Find Best Configurations by Type and Compare
    # =========================================================================
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION COMPARISON")
    print("=" * 80)
    
    # Separate results by model type
    hybrid_lda_results = [r for r in all_results_sorted if any(
        config["name"] == r["Model"] and config.get("type") == "hybrid_lda"
        for config in MODEL_CONFIGURATIONS
    )]
    combined_lda_results = [r for r in all_results_sorted if any(
        config["name"] == r["Model"] and config.get("type") == "combined_adaptive_lda"
        for config in MODEL_CONFIGURATIONS
    )]
    
    # Find best of each type
    best_hybrid_result = hybrid_lda_results[0] if hybrid_lda_results else None
    best_combined_result = combined_lda_results[0] if combined_lda_results else None
    
    best_hybrid_config = None
    best_combined_config = None
    
    if best_hybrid_result:
        best_hybrid_name = best_hybrid_result.get("Model", "")
        for model_config in MODEL_CONFIGURATIONS:
            if model_config["name"] == best_hybrid_name:
                best_hybrid_config = model_config
                break
    
    if best_combined_result:
        best_combined_name = best_combined_result.get("Model", "")
        for model_config in MODEL_CONFIGURATIONS:
            if model_config["name"] == best_combined_name:
                best_combined_config = model_config
                break
    
    # Display best of each type
    print("\n" + "-" * 80)
    print("BEST HYBRIDLDA CONFIGURATION:")
    print("-" * 80)
    if best_hybrid_result and best_hybrid_config:
        print(f"  Model: {best_hybrid_result.get('Model', 'N/A')}")
        print(f"  Accuracy: {best_hybrid_result.get('Acc.', 'N/A')}")
        print(f"  F1 Score: {best_hybrid_result.get('F1 Score', 'N/A')}")
        print(f"  Balanced Acc.: {best_hybrid_result.get('B. Acc.', 'N/A')}")
        print(f"  Configuration: {best_hybrid_config}")
    else:
        print("  No HybridLDA results available")
    
    print("\n" + "-" * 80)
    print("BEST COMBINEDADAPTIVELDA CONFIGURATION:")
    print("-" * 80)
    if best_combined_result and best_combined_config:
        print(f"  Model: {best_combined_result.get('Model', 'N/A')}")
        print(f"  Accuracy: {best_combined_result.get('Acc.', 'N/A')}")
        print(f"  F1 Score: {best_combined_result.get('F1 Score', 'N/A')}")
        print(f"  Balanced Acc.: {best_combined_result.get('B. Acc.', 'N/A')}")
        print(f"  Configuration: {best_combined_config}")
    else:
        print("  No CombinedAdaptiveLDA results available")
    
    # Compare best of each type
    print("\n" + "=" * 80)
    print("BEST COMBINATION COMPARISON (Best HybridLDA vs Best CombinedAdaptiveLDA)")
    print("=" * 80)
    
    if best_hybrid_result and best_combined_result:
        comparison_results = [best_hybrid_result, best_combined_result]
        comparison_table = MetricsTable()
        comparison_table.add_rows(comparison_results)
        comparison_table.display()
        
        # Determine winner
        hybrid_f1 = _f1_sort_key(best_hybrid_result)
        combined_f1 = _f1_sort_key(best_combined_result)
        
        print("\n" + "-" * 80)
        print("WINNER:")
        print("-" * 80)
        if combined_f1 > hybrid_f1:
            print(f"  🏆 CombinedAdaptiveLDA wins! (F1: {combined_f1:.4f} vs {hybrid_f1:.4f})")
            winner_config = best_combined_config
            winner_result = best_combined_result
        elif hybrid_f1 > combined_f1:
            print(f"  🏆 HybridLDA wins! (F1: {hybrid_f1:.4f} vs {combined_f1:.4f})")
            winner_config = best_hybrid_config
            winner_result = best_hybrid_result
        else:
            print(f"  🤝 Tie! (F1: {hybrid_f1:.4f})")
            winner_config = best_combined_config  # Prefer CombinedAdaptiveLDA as default
            winner_result = best_combined_result
        
        # Train and save the overall best model
        print("\n" + "=" * 60)
        print("TRAINING FINAL BEST MODEL")
        print("=" * 60)
        
        X_train_windows, y_train_windows, _ = epochs_to_windows(
            combined_epochs,
            groups,
            window_size=config.window_size,
            step_size=config.step_size,
        )
        
        # Apply artifact removal
        ar_final = ArtefactRemoval()
        ar_final.get_rejection_thresholds(X_train_windows, config)
        X_train_clean, y_train_clean = ar_final.reject_bad_epochs(X_train_windows, y_train_windows)
        
        model_dir = current_wd / "resources" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if winner_config["type"] == "hybrid_lda":
            final_clf = HybridLDAWrapper(
                features=winner_config["features"],
                move_threshold=winner_config["move_threshold"],
                reg=winner_config["reg"],
                shrinkage_alpha=winner_config["shrinkage_alpha"],
                uc_mu=winner_config["uc_mu"],
                sfreq=config.fs,
                use_improved_composition=True
            )
            model_path = model_dir / "hybrid_lda_best.pkl"
        else:  # combined_adaptive_lda
            final_clf = CombinedAdaptiveLDAWrapper(
                features=winner_config["features"],
                confidence_threshold=winner_config.get("confidence_threshold", 0.7),
                ensemble_weight=winner_config.get("ensemble_weight", 0.5),
                move_threshold=winner_config["move_threshold"],
                reg=winner_config["reg"],
                shrinkage_alpha=winner_config["shrinkage_alpha"],
                uc_mu=winner_config["uc_mu"],
                use_adaptive_lr=winner_config.get("use_adaptive_lr", True),
                sfreq=config.fs,
                use_improved_composition=True
            )
            model_path = model_dir / "combined_adaptive_lda_best.pkl"
        
        final_clf.fit(X_train_clean, y_train_clean)
        final_clf.save(str(model_path))
        print(f"Best model ({winner_result.get('Model', 'N/A')}) saved to: {model_path}")
        
        # Store winner info for test set evaluation
        best_config_name = winner_result.get('Model', 'N/A')
    else:
        print("⚠️  Cannot compare: Need both HybridLDA and CombinedAdaptiveLDA results")
        # Use overall best if available
        if all_results_sorted:
            best_overall = all_results_sorted[0]
            best_config_name = best_overall.get('Model', 'N/A')
            # Find config
            for model_config in MODEL_CONFIGURATIONS:
                if model_config["name"] == best_config_name:
                    winner_config = model_config
                    break
            # Train the overall best
            X_train_windows, y_train_windows, _ = epochs_to_windows(
                combined_epochs,
                groups,
                window_size=config.window_size,
                step_size=config.step_size,
            )
            ar_final = ArtefactRemoval()
            ar_final.get_rejection_thresholds(X_train_windows, config)
            X_train_clean, y_train_clean = ar_final.reject_bad_epochs(X_train_windows, y_train_windows)
            model_dir = current_wd / "resources" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            if winner_config["type"] == "hybrid_lda":
                final_clf = HybridLDAWrapper(
                    features=winner_config["features"],
                    move_threshold=winner_config["move_threshold"],
                    reg=winner_config["reg"],
                    shrinkage_alpha=winner_config["shrinkage_alpha"],
                    uc_mu=winner_config["uc_mu"],
                    sfreq=config.fs,
                    use_improved_composition=True
                )
                model_path = model_dir / "hybrid_lda_best.pkl"
            else:
                final_clf = CombinedAdaptiveLDAWrapper(
                    features=winner_config["features"],
                    confidence_threshold=winner_config.get("confidence_threshold", 0.7),
                    ensemble_weight=winner_config.get("ensemble_weight", 0.5),
                    move_threshold=winner_config["move_threshold"],
                    reg=winner_config["reg"],
                    shrinkage_alpha=winner_config["shrinkage_alpha"],
                    uc_mu=winner_config["uc_mu"],
                    use_adaptive_lr=winner_config.get("use_adaptive_lr", True),
                    sfreq=config.fs,
                    use_improved_composition=True
                )
                model_path = model_dir / "combined_adaptive_lda_best.pkl"
            
            final_clf.fit(X_train_clean, y_train_clean)
            final_clf.save(str(model_path))
            print(f"Best model ({best_config_name}) saved to: {model_path}")
        else:
            best_config_name = "N/A"
            final_clf = None

    # =========================================================================
    # 7. Evaluate Final Model on Test Set
    # =========================================================================
    if final_clf is not None and len(x_raw_test) > 0:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)

        # Preprocess test data
        all_test_epochs_list = []
        for raw, events, sub_id, filename in zip(
            x_raw_test, events_test, sub_ids_test, test_filenames
        ):
            # FILTERING: Apply bandpass filter
            filtered_raw = raw.copy()
            filtered_raw.apply_function(filter_obj.apply_filter_offline)

            # CHANNEL REMOVAL: Remove unnecessary channels
            filtered_raw.drop_channels(config.remove_channels)

            # EPOCHING: Create epochs for testing
            epochs = mne.Epochs(
                filtered_raw,
                events,
                event_id=target_event_id,
                tmin=0.3,
                tmax=3.0,
                preload=True,
                baseline=None,
            )

            # Skip files with no epochs
            if len(epochs) == 0:
                print(f"  Skipping {filename}: no epochs after epoching.")
                continue

            all_test_epochs_list.append(epochs)

        if len(all_test_epochs_list) > 0:
            combined_test_epochs = mne.concatenate_epochs(all_test_epochs_list)
            test_labels = combined_test_epochs.events[:, 2]

            # Window test data
            test_groups = np.arange(len(combined_test_epochs))
            X_test_windows, y_test_windows, _ = epochs_to_windows(
                combined_test_epochs,
                test_groups,
                window_size=config.window_size,
                step_size=config.step_size,
            )

            # Artifact removal on test set
            ar_test = ArtefactRemoval()
            ar_test.get_rejection_thresholds(X_test_windows, config)
            X_test_clean, y_test_clean = ar_test.reject_bad_epochs(
                X_test_windows, y_test_windows
            )

            if len(X_test_clean) > 0:
                # Predict on test set
                test_preds = final_clf.predict(X_test_clean)
                test_probs = final_clf.predict_proba(X_test_clean)

                # Compute final metrics
                final_metrics = compile_metrics(
                    y_true=y_test_clean,
                    y_pred=test_preds,
                    y_prob=test_probs,
                    timings=None,
                    n_classes=n_classes,
                )

                print("\n" + "=" * 80)
                print("FINAL TEST SET METRICS")
                print("=" * 80)
                print(f"Test samples: {len(y_test_clean)}")
                print(f"Accuracy:      {final_metrics.get('Acc.', 'N/A')}")
                print(f"Balanced Acc.: {final_metrics.get('B. Acc.', 'N/A')}")
                print(f"F1 Score:     {final_metrics.get('F1 Score', 'N/A')}")
                print(f"ECE:          {final_metrics.get('ECE', 'N/A')}")
                print(f"Brier Score:  {final_metrics.get('Brier', 'N/A')}")
                print("=" * 80)
                print(f"Cross-Validation: {n_folds}-fold (session-wise grouping)")
                print(f"Random State: {config.random_state} (for reproducibility)")
                print("=" * 80)

                # Add to results table
                final_metrics["Model"] = f"{best_config_name} (Test Set)"
                final_metrics_table = MetricsTable()
                final_metrics_table.add_rows([final_metrics])
                print("\nFinal Test Metrics Table:")
                final_metrics_table.display()
            else:
                print("WARNING: No test data remaining after artifact rejection.")
        else:
            print("WARNING: No test epochs created from test files.")
    else:
        print("\nNo test data available for final evaluation.")

    return all_results, df_results


if __name__ == "__main__":
    all_results, df_results = run_hybridlda_comparison_pipeline()  # Function name kept for compatibility

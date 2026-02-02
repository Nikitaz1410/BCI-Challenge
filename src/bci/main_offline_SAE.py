"""
Offline Training and Testing Pipeline for the SAE (Supervised Autoencoder) Model.

This script compares multiple SAE configurations using session-wise grouped
cross-validation.

Pipeline:
1. Load configuration from YAML
2. Load target subject data
3. Filter and preprocess the EEG data
4. Create epochs with metadata for grouped CV (by session)
5. For each model configuration:
   - Perform K-Fold cross-validation
   - Collect metrics
6. Compare all configurations in a summary table

The SAE model uses a convolutional autoencoder for denoising EEG signals,
followed by CSP + classifier for classification. Supported variations:
- Classifier type (csp-lda, csp-svm)
- Mode (supervised)
- Filter configurations

Usage:
    python main_offline_SAE.py
"""

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    create_subject_train_set,
    load_target_subject_data,
)

# Preprocessing
from bci.preprocessing.artefact_removal import ArtefactRemoval
from bci.preprocessing.filters import Filter
from bci.preprocessing.windows import epochs_to_windows, epochs_windows_from_fold

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


def _session_to_day_label(sessions: np.ndarray) -> np.ndarray:
    """
    Convert session strings to integer day labels for SAE.

    The SAE model uses day labels for domain adaptation. Each unique session
    gets a unique integer ID.
    """
    unique_sessions = np.unique(sessions)
    session_to_idx = {s: i for i, s in enumerate(unique_sessions)}
    return np.array([session_to_idx[s] for s in sessions])


def _reject_with_groups(
    ar: ArtefactRemoval,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple:
    """
    Apply artifact rejection while preserving groups alignment.

    Since ArtefactRemoval.reject_bad_epochs doesn't return indices,
    we need to manually track which epochs are kept by replicating
    the rejection logic to get indices.

    Parameters
    ----------
    ar : ArtefactRemoval
        ArtefactRemoval instance with computed thresholds
    X : np.ndarray
        Epoch data (n_epochs, n_channels, n_times)
    y : np.ndarray
        Labels
    groups : np.ndarray
        Group labels (e.g., session IDs)

    Returns
    -------
    tuple
        (X_clean, y_clean, groups_clean)
    """
    from scipy import stats

    n_epochs = X.shape[0]
    n_channels = X.shape[1]
    keep_mask = np.ones(n_epochs, dtype=bool)

    for i in range(n_epochs):
        epoch = X[i]
        bad_channels = np.zeros(n_channels, dtype=bool)

        # Check amplitude
        if ar.use_amplitude and ar.amplitude_threshold is not None:
            channel_max_amps = np.max(np.abs(epoch), axis=1)
            bad_amp = channel_max_amps > ar.amplitude_threshold
            bad_channels = bad_channels | bad_amp
            if np.sum(bad_amp) > (n_channels * ar.max_bad_channels_pct):
                keep_mask[i] = False
                continue

        # Check variance
        if ar.use_variance and ar.variance_threshold is not None:
            channel_vars = np.var(epoch, axis=1)
            bad_var = channel_vars > ar.variance_threshold
            bad_channels = bad_channels | bad_var
            if np.sum(bad_var) > (n_channels * ar.max_bad_channels_pct):
                keep_mask[i] = False
                continue

        # Check gradient
        if ar.use_gradient and ar.gradient_threshold is not None:
            grad = np.diff(epoch, axis=1)
            max_grads = np.max(np.abs(grad), axis=1)
            bad_grad = max_grads > ar.gradient_threshold
            bad_channels = bad_channels | bad_grad
            if np.sum(bad_grad) > (n_channels * ar.max_bad_channels_pct):
                keep_mask[i] = False
                continue

        # Check consistency
        if ar.use_consistency and ar.consistency_threshold is not None:
            channel_vars = np.var(epoch, axis=1)
            if len(channel_vars) > 2:
                z_scores = np.abs(stats.zscore(channel_vars))
                if np.max(z_scores) > ar.consistency_threshold:
                    keep_mask[i] = False
                    continue

        # Final check: too many bad channels overall
        if np.sum(bad_channels) > (n_channels * ar.max_bad_channels_pct):
            keep_mask[i] = False

    return X[keep_mask], y[keep_mask], groups[keep_mask]


# =============================================================================
# Model Configurations to Compare
# =============================================================================
MODEL_CONFIGURATIONS: List[Dict[str, Any]] = [
    # SAE + CSP-LDA (Default)
    {
        "name": "SAE + CSP-LDA",
        "mode": "supervised",
        "classifier": "csp-lda",
        "filters": [8, 16, 32],
    },
    # SAE + CSP-SVM
    {
        "name": "SAE + CSP-SVM",
        "mode": "supervised",
        "classifier": "csp-svm",
        "filters": [8, 16, 32],
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
    use_artifact_rejection: bool = True,
) -> Dict[str, Any]:
    """
    Run cross-validation for a single SAE model configuration.

    Parameters
    ----------
    model_config : dict
        Model configuration with 'name', 'mode', 'classifier', 'filters'
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
    use_artifact_rejection : bool
        Whether to apply artifact rejection (default True for SAE)

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
        groups_train_fold = fold_windowed_epochs["groups_train"]
        groups_val_fold = fold_windowed_epochs["groups_val"]

        # Artifact removal within each fold (optional for SAE)
        if use_artifact_rejection:
            ar = ArtefactRemoval()
            ar.get_rejection_thresholds(X_train_fold, config)

            # Reject bad epochs (with groups handled via helper)
            X_train_clean, y_train_clean, groups_train_clean = _reject_with_groups(
                ar, X_train_fold, y_train_fold, groups_train_fold
            )

            ar_val = ArtefactRemoval()
            ar_val.get_rejection_thresholds(X_val_fold, config)
            X_val_clean, y_val_clean, groups_val_clean = _reject_with_groups(
                ar_val, X_val_fold, y_val_fold, groups_val_fold
            )

            # Skip if no data left after artifact rejection
            if len(X_train_clean) == 0 or len(X_val_clean) == 0:
                print("Skipped (no data after AR)")
                continue
        else:
            X_train_clean, y_train_clean = X_train_fold, y_train_fold
            X_val_clean, y_val_clean = X_val_fold, y_val_fold
            groups_train_clean = groups_train_fold
            groups_val_clean = groups_val_fold

        # Convert session groups to integer day labels for SAE
        day_labels_train = _session_to_day_label(groups_train_clean)
        day_labels_val = _session_to_day_label(groups_val_clean)

        # Create and train the SAE model
        clf = choose_model(
            "sae",
            {
                "mode": model_config["mode"],
                "classifier": model_config["classifier"],
                "filters": model_config["filters"],
            },
        )

        try:
            clf.fit(X_train_clean, y_train_clean, day_labels=day_labels_train)

            # Evaluate on validation fold
            fold_predictions = clf.predict(X_val_clean, day_labels=day_labels_val)
            fold_probabilities = clf.predict_proba(X_val_clean, day_labels=day_labels_val)

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


def run_sae_comparison_pipeline(
    configurations: Optional[List[Dict[str, Any]]] = None,
    use_artifact_rejection: bool = True,
):
    """
    Main pipeline that compares multiple SAE configurations using
    session-wise grouped cross-validation.

    Parameters
    ----------
    configurations : list, optional
        List of model configurations to evaluate. If None, uses MODEL_CONFIGURATIONS.
    use_artifact_rejection : bool
        Whether to apply artifact rejection (default True for SAE)
    """
    # Use default configurations if none provided
    if configurations is None:
        configurations = MODEL_CONFIGURATIONS

    # =========================================================================
    # 1. Load Configuration
    # =========================================================================
    current_wd = Path.cwd()  # BCI-Challenge directory

    try:
        config_path = current_wd / "resources" / "configs" / "bci_config.yaml"
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        print("Configuration loaded successfully!")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Initialize variables
    np.random.seed(config.random_state)

    # n_folds will be set to number of sessions after loading data

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
            num_general=0,
            num_dino=13,
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

    print(f"Using {n_folds}-fold cross-validation (one fold per session, grouped by session).")
    print(f"Training data shape: {X_train.shape}")
    print(f"Labels distribution: {np.unique(y_train, return_counts=True)}")
    print(f"Number of sessions (CV groups): {len(unique_sessions)}")
    print(f"Sessions: {list(unique_sessions)}")

    # =========================================================================
    # 4. Run Cross-Validation for Each Configuration
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPARING SAE CONFIGURATIONS")
    print("=" * 60)
    print(f"Total configurations to evaluate: {len(configurations)}")
    print(f"Artifact rejection: {'Enabled' if use_artifact_rejection else 'Disabled'}")

    all_results = []
    n_classes = len(target_event_id)

    for config_idx, model_config in enumerate(configurations):
        print(f"\n[{config_idx + 1}/{len(configurations)}] ", end="")

        result = run_cv_for_config(
            model_config=model_config,
            combined_epochs=combined_epochs,
            groups=groups,
            X_train=X_train,
            y_train=y_train,
            config=config,
            n_classes=n_classes,
            n_folds=n_folds,
            use_artifact_rejection=use_artifact_rejection,
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

    all_results_sorted = sorted(all_results, key=_f1_sort_key, reverse=True)

    print("\n" + "=" * 80)
    print("SAE MODEL COMPARISON RESULTS (sorted by F1 Score, descending)")
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
        for model_config in configurations:
            if model_config["name"] == best_config_name:
                best_config = model_config
                break

    if best_config:
        print(f"Best model: {best_config_name}")
        print(f"Mean CV F1 Score: {best_f1:.4f}")
        print(f"Configuration: {best_config}")

        # Train final model with best configuration on all data
        print("\nTraining final model with best configuration on all data...")

        X_train_windows, y_train_windows, groups_windows = epochs_to_windows(
            combined_epochs,
            groups,
            window_size=config.window_size,
            step_size=config.step_size,
        )

        # Apply artifact rejection if enabled
        if use_artifact_rejection:
            ar = ArtefactRemoval()
            ar.get_rejection_thresholds(X_train_windows, config)
            X_train_windows, y_train_windows, groups_windows = _reject_with_groups(
                ar, X_train_windows, y_train_windows, groups_windows
            )

        # Convert session groups to integer day labels
        day_labels_train = _session_to_day_label(groups_windows)

        print(f"Training on {len(X_train_windows)} windows...")

        final_clf = choose_model(
            "sae",
            {
                "mode": best_config["mode"],
                "classifier": best_config["classifier"],
                "filters": best_config["filters"],
            },
        )

        final_clf.fit(X_train_windows, y_train_windows, day_labels=day_labels_train)

        # Save the best model
        model_dir = current_wd / "resources" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "sae_best_model.pt"
        final_clf.save(str(model_path))
        print(f"Best model saved to: {model_path}")

    return all_results, df_results


def run_single_sae_config(
    mode: str = "supervised",
    classifier: str = "csp-lda",
    use_artifact_rejection: bool = True,
):
    """
    Run training and evaluation for a single SAE configuration.

    This is a convenience function for users who want to run a single
    configuration without comparing multiple settings.

    Parameters
    ----------
    mode : str
        SAE mode (default: 'supervised')
    classifier : str
        Classifier to use with CSP features: 'csp-lda' or 'csp-svm' (default: 'csp-lda')
    use_artifact_rejection : bool
        Whether to apply artifact rejection (default True)

    Note
    ----
    The SAE model uses fixed filter configuration [8, 16, 32] due to
    tight coupling between latent_sz calculation and classifier dimensions.
    """
    # Fixed filters - SAE model architecture requires this specific configuration
    filters = [8, 16, 32]

    single_config = [
        {
            "name": f"SAE ({mode}, {classifier})",
            "mode": mode,
            "classifier": classifier,
            "filters": filters,
        }
    ]

    return run_sae_comparison_pipeline(
        configurations=single_config,
        use_artifact_rejection=use_artifact_rejection,
    )


if __name__ == "__main__":
    # Run the full comparison pipeline
    all_results, df_results = run_sae_comparison_pipeline()

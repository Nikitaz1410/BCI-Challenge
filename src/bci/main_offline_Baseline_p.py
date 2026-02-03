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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupKFold, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

# Feature extraction (CSP) for classic ML
from bci.models.Baseline import CSPFeatureExtractor

# Utils
from bci.utils.bci_config import load_config


class CSPSVMEstimator(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible estimator that combines:
    - CSP feature extraction (multi-class, one-vs-rest)
    - Standardization
    - RBF-kernel SVM

    This is used as the base estimator in the inner CV hyperparameter search.
    """

    def __init__(
        self,
        n_csp_components: int = 6,
        C: float = 1.0,
        gamma: Any = "scale",
        random_state: int = 42,
    ) -> None:
        self.n_csp_components = n_csp_components
        self.C = C
        self.gamma = gamma
        self.random_state = random_state

        # Fitted components
        self._csp: CSPFeatureExtractor | None = None
        self._scaler: StandardScaler | None = None
        self._svm: SVC | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPSVMEstimator":
        # X has shape (n_epochs, n_channels, n_times)
        csp = CSPFeatureExtractor(
            n_csp_components=int(self.n_csp_components),
            use_multiclass_csp=True,
        )
        features = csp.fit_transform(X, y)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        svm = SVC(
            kernel="rbf",
            C=float(self.C),
            gamma=self.gamma,
            probability=True,
            class_weight="balanced",
            random_state=self.random_state,
        )
        svm.fit(features_scaled, y)

        self._csp = csp
        self._scaler = scaler
        self._svm = svm
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._csp is None or self._scaler is None:
            raise RuntimeError("CSPSVMEstimator must be fitted before calling predict/predict_proba.")
        feats = self._csp.transform(X)
        return self._scaler.transform(feats)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._svm is None:
            raise RuntimeError("CSPSVMEstimator must be fitted before calling predict.")
        X_trans = self._transform(X)
        return self._svm.predict(X_trans)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._svm is None:
            raise RuntimeError("CSPSVMEstimator must be fitted before calling predict_proba.")
        X_trans = self._transform(X)
        return self._svm.predict_proba(X_trans)


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


MODEL_NAME = "CSP + SVM (tuned)"


def run_cv_for_config(
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
    config_name = MODEL_NAME
    print(f"\n{'='*60}")
    print(f"Evaluating CSP + SVM with hyperparameter tuning")
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
            # ------------------------------------------------------------------
            # Inner CV: hyperparameter search for CSP + SVM on the training fold
            # ------------------------------------------------------------------
            # Determine a safe number of inner folds based on class counts
            class_counts = np.bincount(y_train_clean)
            max_possible_splits = int(class_counts[class_counts > 0].min())
            n_inner_splits = max(2, min(3, max_possible_splits))

            inner_cv = StratifiedKFold(
                n_splits=n_inner_splits,
                shuffle=True,
                random_state=config.random_state,
            )

            param_grid = {
                "n_csp_components": [4, 6, 8],
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", "auto"],
            }

            base_estimator = CSPSVMEstimator(random_state=config.random_state)

            grid_search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=inner_cv,
                n_jobs=1,
                refit=True,
            )

            grid_search.fit(X_train_clean, y_train_clean)
            best_clf: CSPSVMEstimator = grid_search.best_estimator_

            print(
                f"best params fold {fold_idx + 1}: "
                f"{grid_search.best_params_}",
                end="; ",
            )

            # Evaluate tuned model on validation fold
            fold_predictions = best_clf.predict(X_val_clean)
            fold_probabilities = best_clf.predict_proba(X_val_clean)

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
            num_general=6,
            num_dino=18,
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
    print("CSP + SVM HYPERPARAMETER TUNING (LOSO CV)")
    print("=" * 60)

    all_results = []
    n_classes = len(target_event_id)

    # Single configuration: CSP + SVM with tuned hyperparameters
    result = run_cv_for_config(
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
    # 6. Train Final CSP + SVM Model with Tuned Hyperparameters
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING FINAL CSP + SVM MODEL (tuned on all data)")
    print("=" * 60)

    X_train_windows, y_train_windows, _ = epochs_to_windows(
        combined_epochs,
        groups,
        window_size=config.window_size,
        step_size=config.step_size,
    )

    # Artifact removal (same as in CV)
    ar = ArtefactRemoval()
    ar.get_rejection_thresholds(X_train_windows, config)
    X_train_clean, y_train_clean = ar.reject_bad_epochs(
        X_train_windows, y_train_windows
    )

    if len(X_train_clean) == 0:
        print("Warning: No data left after artifact rejection. Skipping final model save.")
    else:
        # Inner CV on all available training data
        class_counts_full = np.bincount(y_train_clean)
        max_possible_splits_full = int(class_counts_full[class_counts_full > 0].min())
        n_inner_splits_full = max(2, min(3, max_possible_splits_full))

        inner_cv_full = StratifiedKFold(
            n_splits=n_inner_splits_full,
            shuffle=True,
            random_state=config.random_state,
        )

        param_grid_full = {
            "n_csp_components": [4, 6, 8],
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto"],
        }

        base_estimator_full = CSPSVMEstimator(random_state=config.random_state)

        grid_search_full = GridSearchCV(
            estimator=base_estimator_full,
            param_grid=param_grid_full,
            scoring="f1_macro",
            cv=inner_cv_full,
            n_jobs=1,
            refit=True,
        )

        grid_search_full.fit(X_train_clean, y_train_clean)
        final_clf: CSPSVMEstimator = grid_search_full.best_estimator_

        print(f"Best hyperparameters on full data: {grid_search_full.best_params_}")

        # Save the tuned model and ArtefactRemoval
        model_dir = current_wd / "resources" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "csp_svm_tuned.pkl"
        artefact_path = model_dir / "artefact_removal.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(final_clf, f)
        with open(artefact_path, "wb") as f:
            pickle.dump(ar, f)

        print(f"Tuned CSP+SVM model saved to: {model_path}")
        print(f"ArtefactRemoval saved to: {artefact_path}")

    return all_results, df_results


if __name__ == "__main__":
    all_results, df_results = run_baseline_comparison_pipeline()

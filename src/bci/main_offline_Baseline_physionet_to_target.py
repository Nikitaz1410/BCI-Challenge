"""
Offline transfer-learning pipeline for the Baseline (AllRounder) models.

This script:
1. Loads configuration from YAML
2. Loads **all** Physionet Motor Imagery data (multi-subject)
3. Loads **all** available target-subject data (e.g. subject 110)
4. Applies the same preprocessing as `main_offline_Baseline.py`
5. Trains the **same Baseline model configurations** on Physionet
6. Evaluates each trained model on all target-subject data
7. Prints a comparison table of metrics on the target subject

Usage:
    python main_offline_Baseline_physionet_to_target.py
"""

from __future__ import annotations

import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add src directory to Python path to allow imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import mne
import numpy as np
import pandas as pd

# Evaluation
from bci.evaluation.metrics import MetricsTable, compile_metrics

# Data Acquisition
from bci.loading.loading import (
    load_physionet_data,
    load_target_subject_data,
)

# Preprocessing
from bci.preprocessing.artefact_removal import ArtefactRemoval
from bci.preprocessing.filters import Filter
from bci.preprocessing.windows import epochs_to_windows

# Models - HybridLDA wrapper for testing different features
from bci.models.adaptive_lda_modules.hybrid_lda_wrapper import HybridLDAWrapper

# Utils
from bci.utils.bci_config import load_config
from bci.utils.utils import choose_model


def _session_id_from_filename(filename: str) -> str:
    """
    Extract session identifier from BIDS-style filename for grouping.

    E.g. "sub-P999_ses-S002_task-dino_run-001_eeg_raw" -> "sub-P999_ses-S002".
    If the pattern is not found, returns the full filename.
    """
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
    # CSP + Random Forest
    {
        "name": "CSP + RF",
        "features": "csp",
        "classifier": "rf",
        "scale": True,
    },
    # Band power + LDA
    {
        "name": "BP + LDA",
        "features": "welch_bandpower",
        "classifier": "lda",
        "scale": True,
    },
    # Band power + SVM
    {
        "name": "BP + SVM",
        "features": "welch_bandpower",
        "classifier": "svm",
        "scale": True,
    },
    # Band power + Logistic Regression
    {
        "name": "BP + LogReg",
        "features": "welch_bandpower",
        "classifier": "logreg",
        "scale": True,
    },
    # Band power + Random Forest
    {
        "name": "BP + RF",
        "features": "welch_bandpower",
        "classifier": "rf",
        "scale": True,
    },
    # CSP + Band power + LDA (combined features)
    {
        "name": "CSP & BP + LDA",
        "features": ["csp", "welch_bandpower"],
        "classifier": "lda",
        "scale": True,
    },
    # CSP + Band power + SVM
    {
        "name": "CSP & BP + SVM",
        "features": ["csp", "welch_bandpower"],
        "classifier": "svm",
        "scale": True,
    },
    # CSP + Band power + Logistic Regression
    {
        "name": "CSP & BP + LogReg",
        "features": ["csp", "welch_bandpower"],
        "classifier": "logreg",
        "scale": True,
    },
    # CSP + Band power + Random Forest
    {
        "name": "CSP & BP + RF",
        "features": ["csp", "welch_bandpower"],
        "classifier": "rf",
        "scale": True,
    },
]


def _prepare_epochs_from_raws(
    raws: List[mne.io.BaseRaw],
    events_list: List[np.ndarray],
    event_id: Dict[str, int],
    sub_ids: List[int],
    filenames: List[str],
    filter_obj: Filter,
    remove_channels: List[str],
    tmin: float = 0.3,
    tmax: float = 3.0,
) -> Tuple[mne.Epochs, np.ndarray]:
    """
    Common preprocessing: filter, drop channels, epoch, attach metadata.

    Returns
    -------
    combined_epochs : mne.Epochs
        Concatenated epochs across all inputs.
    groups : np.ndarray
        Group labels (here: session IDs) for each epoch.
    """
    all_epochs_list: List[mne.Epochs] = []

    for raw, events, sub_id, filename in zip(raws, events_list, sub_ids, filenames):
        filtered_raw = raw.copy()
        filtered_raw.apply_function(filter_obj.apply_filter_offline)

        if remove_channels:
            existing = [ch for ch in remove_channels if ch in filtered_raw.ch_names]
            if existing:
                filtered_raw.drop_channels(existing)

        epochs = mne.Epochs(
            filtered_raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
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
        raise ValueError(
            "No epochs after preprocessing. Check events/event_id and epoching window."
        )

    combined_epochs = mne.concatenate_epochs(all_epochs_list)
    groups = combined_epochs.metadata["session"].values
    return combined_epochs, groups


def _evaluate_transfer_for_config(
    model_config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Any,
    n_classes: int,
) -> Dict[str, Any]:
    """
    Train a single configuration on Physionet and evaluate on target subject.
    """
    config_name = model_config["name"]
    print(f"\n{'='*60}")
    print(f"Training on Physionet, evaluating on target: {config_name}")
    print(f"{'='*60}")

    # Artefact removal: thresholds from Physionet (train) only
    ar = ArtefactRemoval()
    ar.get_rejection_thresholds(X_train, config)
    X_train_clean, y_train_clean = ar.reject_bad_epochs(X_train, y_train)
    X_test_clean, y_test_clean = ar.reject_bad_epochs(X_test, y_test)

    if len(X_train_clean) == 0:
        print("  Skipped (no training data after AR)")
        return {
            "Model": config_name,
            "Acc.": "N/A",
            "B. Acc.": "N/A",
            "F1 Score": "N/A",
            "ECE": "N/A",
            "Brier": "N/A",
            "Train Time (s)": "N/A",
            "Avg. Filter Latency (ms)": "N/A",
            "Avg. Infer Latency (ms)": "N/A",
            "Avg. Total Latency (ms)": "N/A",
        }

    if len(X_test_clean) == 0:
        print("  Skipped (no test data after AR)")
        return {
            "Model": config_name,
            "Acc.": "N/A",
            "B. Acc.": "N/A",
            "F1 Score": "N/A",
            "ECE": "N/A",
            "Brier": "N/A",
            "Train Time (s)": "N/A",
            "Avg. Filter Latency (ms)": "N/A",
            "Avg. Infer Latency (ms)": "N/A",
            "Avg. Total Latency (ms)": "N/A",
        }

    start_time = time.perf_counter()
    timings = dict()
    timings["filter_latency"] = 89.48  # Avg. Filter Group Delay

    try:
        if model_config["classifier"] == "hybrid_lda":
            clf = HybridLDAWrapper(
                features=model_config["features"],
                move_threshold=0.5,
                reg=1e-2,
                shrinkage_alpha=0.1,
                uc_mu=0.4 * 2**-6,
                sfreq=config.fs,
            )
            start_train_time = time.perf_counter()
            clf.fit(X_train_clean, y_train_clean)
            timings["train_time"] = time.perf_counter() - start_train_time
            start_pred_time = time.perf_counter()
            y_pred = clf.predict(X_test_clean)
            y_prob = clf.predict_proba(X_test_clean)
        else:
            clf = choose_model(
                "baseline",
                {
                    "features": model_config["features"],
                    "classifier": model_config["classifier"],
                    "scale": model_config["scale"],
                    "random_state": config.random_state,
                },
            )
            start_train_time = time.perf_counter()
            clf.fit(X_train_clean, y_train_clean)
            timings["train_time"] = time.perf_counter() - start_train_time
            start_pred_time = time.perf_counter()
            y_pred = clf.predict(X_test_clean)
            y_prob = clf.predict_proba(X_test_clean)

        timings["infer_latency"] = (
            time.perf_counter() - start_pred_time
        ) * 1000 / X_test_clean.shape[0] + (
            0.07 + 0.02
        )  # Avg. Filtering and AR based on real-time computation on M2 chip
        timings["total_latency"] = (
            timings["infer_latency"] * 10 + 89.48
        )  # TransferFunction Number of Classification for Buffer, Avg. Filter Group Delay

        metrics = compile_metrics(
            y_true=y_test_clean,
            y_pred=y_pred,
            y_prob=y_prob,
            timings=timings,
            n_classes=n_classes,
        )

        elapsed = time.perf_counter() - start_time
        print(
            f"  Target Acc: {metrics['Acc.']:.4f}, "
            f"F1: {metrics['F1 Score']:.4f} ({elapsed:.1f}s)"
        )

        result = {
            "Model": config_name,
            "Acc.": f"{metrics['Acc.']:.4f}",
            "B. Acc.": f"{metrics['B. Acc.']:.4f}",
            "F1 Score": f"{metrics['F1 Score']:.4f}",
            "ECE": f"{metrics['ECE']:.4f}",
            "Brier": f"{metrics['Brier']:.4f}",
            "Eval Time (s)": f"{elapsed:.1f}",
        }
        if "Train Time (s)" in metrics:
            result["Train Time (s)"] = f"{metrics['Train Time (s)']:.2f}"
            result["Avg. Filter Latency (ms)"] = f"{metrics['Avg. Filter Latency (ms)']:.2f}"
            result["Avg. Infer Latency (ms)"] = f"{metrics['Avg. Infer Latency (ms)']:.2f}"
            result["Avg. Total Latency (ms)"] = f"{metrics['Avg. Total Latency (ms)']:.2f}"
        return result

    except Exception as e:
        print(f"  Error during training/evaluation: {e}")
        return {
            "Model": config_name,
            "Acc.": "N/A",
            "B. Acc.": "N/A",
            "F1 Score": "N/A",
            "ECE": "N/A",
            "Brier": "N/A",
            "Train Time (s)": "N/A",
            "Avg. Filter Latency (ms)": "N/A",
            "Avg. Infer Latency (ms)": "N/A",
            "Avg. Total Latency (ms)": "N/A",
        }


def run_physionet_to_target_baseline_pipeline():
    """
    Train Baseline models on Physionet and evaluate on target subject data.
    """
    # -------------------------------------------------------------------------
    # 1. Load configuration and set up paths
    # -------------------------------------------------------------------------
    script_dir = Path(__file__).parent.parent.parent

    if (script_dir / "src").exists() and (script_dir / "data").exists():
        current_wd = script_dir
    elif (script_dir / "BCI-Challenge" / "src").exists() and (
        script_dir / "BCI-Challenge" / "data"
    ).exists():
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

    random.seed(config.random_state)
    np.random.seed(config.random_state)

    filter_obj = Filter(config, online=False)

    # -------------------------------------------------------------------------
    # 2. Load Physionet data (training) and target subject data (evaluation)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING PHYSIONET DATA (TRAIN)")
    print("=" * 60)

    # Use all Physionet subjects specified in config.subjects
    physio_subjects = config.subjects
    print(f"Physionet subjects for training: {physio_subjects}")

    (
        physio_raws,
        physio_events,
        physio_event_id,
        physio_subject_ids,
        physio_filenames,
    ) = load_physionet_data(
        subjects=physio_subjects,
        root=current_wd,
        channels=config.channels,
    )

    print(f"Loaded {len(physio_raws)} Physionet subjects.")

    print("\n" + "=" * 60)
    print("LOADING TARGET SUBJECT DATA (EVAL)")
    print("=" * 60)

    target_source_path = current_wd / "data" / "eeg" / config.target
    target_dataset_path = current_wd / "data" / "datasets" / config.target

    (
        target_raws,
        target_events,
        target_event_id,
        target_sub_ids,
        target_metadata,
    ) = load_target_subject_data(
        root=current_wd,
        source_path=target_source_path,
        target_path=target_dataset_path,
        resample=None,
    )

    print(f"Loaded {len(target_raws)} sessions from target subject data.")

    # -------------------------------------------------------------------------
    # 3. Preprocessing: filter + epoch Physionet and target data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPROCESSING PHYSIONET DATA (TRAIN)")
    print("=" * 60)

    physio_epochs, physio_groups = _prepare_epochs_from_raws(
        physio_raws,
        physio_events,
        physio_event_id,
        physio_subject_ids,
        physio_filenames,
        filter_obj=filter_obj,
        remove_channels=config.remove_channels or [],
        tmin=0.3,
        tmax=3.0,
    )

    X_physio = physio_epochs.get_data()
    y_physio = physio_epochs.events[:, 2]
    print(f"Physionet epochs shape: {X_physio.shape}")
    print(f"Physionet label distribution: {np.unique(y_physio, return_counts=True)}")

    print("\n" + "=" * 60)
    print("PREPROCESSING TARGET DATA (EVAL)")
    print("=" * 60)

    target_filenames = target_metadata["filenames"]
    target_epochs, target_groups = _prepare_epochs_from_raws(
        target_raws,
        target_events,
        target_event_id,
        target_sub_ids,
        target_filenames,
        filter_obj=filter_obj,
        remove_channels=config.remove_channels or [],
        tmin=0.3,
        tmax=3.0,
    )

    X_target = target_epochs.get_data()
    y_target = target_epochs.events[:, 2]
    print(f"Target epochs shape: {X_target.shape}")
    print(f"Target label distribution: {np.unique(y_target, return_counts=True)}")

    # -------------------------------------------------------------------------
    # 4. Windowing for Baseline models
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CREATING WINDOWS")
    print("=" * 60)

    X_train_windows, y_train_windows, _ = epochs_to_windows(
        physio_epochs,
        physio_groups,
        window_size=config.window_size,
        step_size=config.step_size,
    )

    X_test_windows, y_test_windows, _ = epochs_to_windows(
        target_epochs,
        target_groups,
        window_size=config.window_size,
        step_size=config.step_size,
    )

    print(f"Train windows shape (Physionet): {X_train_windows.shape}")
    print(f"Test windows shape (Target): {X_test_windows.shape}")

    n_classes = len(target_event_id)

    # -------------------------------------------------------------------------
    # 5. Train each configuration on Physionet and evaluate on target
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRANSFER BASELINE MODEL COMPARISON (Physionet -> Target)")
    print("=" * 60)
    print(f"Total configurations to evaluate: {len(MODEL_CONFIGURATIONS)}")

    all_results: List[Dict[str, Any]] = []

    for idx, model_config in enumerate(MODEL_CONFIGURATIONS):
        print(f"\n[{idx + 1}/{len(MODEL_CONFIGURATIONS)}] {model_config['name']}")
        result = _evaluate_transfer_for_config(
            model_config=model_config,
            X_train=X_train_windows,
            y_train=y_train_windows,
            X_test=X_test_windows,
            y_test=y_test_windows,
            config=config,
            n_classes=n_classes,
        )
        all_results.append(result)

    # -------------------------------------------------------------------------
    # 6. Sort by F1 Score and display comparison
    # -------------------------------------------------------------------------
    def _f1_sort_key(result: Dict[str, Any]) -> float:
        f1_str = result.get("F1 Score", "N/A")
        if f1_str == "N/A":
            return -1.0
        try:
            return float(str(f1_str).split("+/-")[0].strip())
        except (ValueError, IndexError):
            try:
                return float(str(f1_str).strip())
            except ValueError:
                return -1.0

    all_results_sorted = sorted(all_results, key=_f1_sort_key, reverse=True)

    print("\n" + "=" * 80)
    print("TRANSFER LEARNING RESULTS (Physionet train -> Target eval)")
    print("=" * 80)
    print(f"Total Physionet epochs: {len(X_physio)}")
    print(f"Total target epochs: {len(X_target)}")
    print(f"Total train windows: {len(X_train_windows)}")
    print(f"Total test windows: {len(X_test_windows)}")
    print("=" * 80)

    comparison_table = MetricsTable()
    comparison_table.add_rows(all_results_sorted)
    comparison_table.display()

    df_results = pd.DataFrame(all_results_sorted)

    # Optionally save best model (trained on all Physionet + AR on windows)
    if all_results_sorted and _f1_sort_key(all_results_sorted[0]) >= 0:
        best_name = all_results_sorted[0]["Model"]
        best_config = next(
            (mc for mc in MODEL_CONFIGURATIONS if mc["name"] == best_name), None
        )
        if best_config is not None:
            print("\n" + "=" * 60)
            print("TRAINING FINAL BEST MODEL ON ALL PHYSIONET WINDOWS")
            print("=" * 60)

            ar_final = ArtefactRemoval()
            ar_final.get_rejection_thresholds(X_train_windows, config)
            X_train_clean, y_train_clean = ar_final.reject_bad_epochs(
                X_train_windows, y_train_windows
            )

            if len(X_train_clean) == 0:
                print(
                    "Warning: No Physionet data left after artefact removal. "
                    "Skipping final model save."
                )
            else:
                model_dir = current_wd / "resources" / "models"
                model_dir.mkdir(parents=True, exist_ok=True)

                if best_config["classifier"] == "hybrid_lda":
                    final_clf = HybridLDAWrapper(
                        features=best_config["features"],
                        move_threshold=0.5,
                        reg=1e-2,
                        shrinkage_alpha=0.1,
                        uc_mu=0.4 * 2**-6,
                        sfreq=config.fs,
                    )
                    final_clf.fit(X_train_clean, y_train_clean)
                    model_path = model_dir / "hybrid_lda_physionet_best.pkl"
                    final_clf.save(str(model_path))
                    print(f"Best HybridLDA Physionet model saved to: {model_path}")
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

                    model_path = model_dir / "baseline_physionet_best_model.pkl"
                    artefact_path = model_dir / "artefact_removal_physionet.pkl"

                    final_clf.save(str(model_path))
                    with open(artefact_path, "wb") as f:
                        pickle.dump(ar_final, f)

                    print(f"Best Physionet-trained model saved to: {model_path}")
                    print(f"ArtefactRemoval (Physionet) saved to: {artefact_path}")

    return all_results_sorted, df_results


if __name__ == "__main__":
    all_results, df_results = run_physionet_to_target_baseline_pipeline()


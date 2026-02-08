"""
Offline Training and Testing Pipeline for the Baseline (AllRounder) Model.
Implements session-wise GroupKFold Cross-Validation for Riemannian Geometry.
"""

import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Local BCI Modules
from bci.evaluation.metrics import MetricsTable, compile_metrics
from bci.loading.loading import (
    create_subject_train_set,
    load_target_subject_data,
)
from bci.models.riemann import RiemannianClf
from bci.preprocessing.artefact_removal import ArtefactRemoval
from bci.preprocessing.filters import Filter
from bci.preprocessing.windows import epochs_to_windows, epochs_windows_from_fold
from bci.utils.bci_config import load_config


def _session_id_from_filename(filename: str) -> str:
    """Extracts BIDS-style session identifier for CV grouping."""
    base = filename.replace("_raw", "").strip("_")
    match = re.match(r"(sub-[^_]+_ses-[^_]+)", base)
    return match.group(1) if match else filename


def run_cv_for_config(
    combined_epochs: mne.Epochs,
    groups: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Any,
    n_classes: int,
) -> Dict[str, Any]:
    """Runs GroupKFold CV for the Riemannian Classifier."""

    config_name = "RiemannianClf"
    print(f"\n{'-'*60}\nEvaluating: {config_name}\n{'-'*60}")

    # Use leave-one-session-out CV
    n_folds = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_folds)
    cv_metrics_list, fold_times = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X_train, y_train, groups=groups)
    ):
        fold_start = time.perf_counter()
        print(f"  Fold {fold_idx + 1}/{n_folds}...", end=" ")

        # 1. Windowing
        fold_windows = epochs_windows_from_fold(
            combined_epochs,
            groups,
            train_idx,
            val_idx,
            window_size=config.window_size,
            step_size=config.step_size,
        )

        # 2. Artifact Removal
        ar = ArtefactRemoval()
        ar.get_rejection_thresholds(fold_windows["X_train"], config)

        X_tr_clean, y_tr_clean, gr_tr_clean = ar.reject_bad_epochs_riemann(
            fold_windows["X_train"],
            fold_windows["y_train"],
            fold_windows["groups_train"],
        )
        X_val_clean, y_val_clean, _ = ar.reject_bad_epochs_riemann(
            fold_windows["X_val"], fold_windows["y_val"]
        )

        if len(X_tr_clean) == 0 or len(X_val_clean) == 0:
            print("Skipped (No data post-AR)")
            continue

        # 3. Fit and Predict
        clf = RiemannianClf(cov_est="lwf")
        t_fit_start = time.perf_counter()
        # clf.fit(X_tr_clean, y_tr_clean)
        clf.fit_centered(X_tr_clean, y_tr_clean, gr_tr_clean)
        # clf.fit_special(X_tr_clean, y_tr_clean, gr_tr_clean)
        train_time = time.perf_counter() - t_fit_start

        t_inf_start = time.perf_counter()
        # COMMENT when using fit_centered() or fit_special()
        # preds, _ = clf.predict(X_val_clean)
        # probs, _ = clf.predict_proba(X_val_clean)

        # UNCOMMENT COMMENT when using fit_centered() or fit_special()
        preds, probs, _ = clf.predict_with_recentering(X_val_clean)

        # Calculate AR/Filtering Effort (M2 Chip baseline constants)
        inf_latency = (
            (time.perf_counter() - t_inf_start) * 1000 / X_val_clean.shape[0]
        ) + 0.09

        timings = {
            "train_time": train_time,
            "filter_latency": 89.48,
            "infer_latency": inf_latency,
            "total_latency": (inf_latency * 10) + 89.48,
        }

        # 4. Metrics
        fold_metrics = compile_metrics(y_val_clean, preds, probs, timings, n_classes)
        cv_metrics_list.append(fold_metrics)

        fold_duration = time.perf_counter() - fold_start
        fold_times.append(fold_duration)
        print(f"Acc: {fold_metrics['Acc.']:.4f} ({fold_duration:.1f}s)")

    return _aggregate_results(config_name, cv_metrics_list, fold_times)


def _aggregate_results(
    name: str, metrics: List[Dict], fold_times: List[float]
) -> Dict[str, Any]:
    """Helper to compute mean/std across CV folds."""
    if not metrics:
        return {"Model": name, "Acc.": "N/A"}

    res = {"Model": name}
    keys = [
        "Acc.",
        "B. Acc.",
        "F1 Score",
        "ECE",
        "Brier",
        "Train Time (s)",
        "Avg. Filter Latency (ms)",
        "Avg. Infer Latency (ms)",
        "Avg. Total Latency (ms)",
    ]

    for key in keys:
        vals = [m[key] for m in metrics if key in m]
        res[key] = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}" if vals else "N/A"

    res["Avg. Fold Time (s)"] = f"{np.mean(fold_times):.1f}"
    return res


def run_baseline_comparison_pipeline():
    """Main Orchestrator for the BCI Training Pipeline."""
    root = Path.cwd()

    # 1. Setup Config
    config = load_config(root / "resources" / "configs" / "bci_config.yaml")
    np.random.seed(config.random_state)

    # 2. Data Acquisition
    print(f"\n{'='*60}\nLOADING DATA: {config.target}\n{'='*60}")
    raws, events, evt_id, _, meta = load_target_subject_data(
        root=root,
        source_path=root / "data" / "eeg" / config.target,
        target_path=root / "data" / "datasets" / config.target,
        resample=False,
    )

    x_train_raw, y_train_evt, filenames, sub_ids, _ = create_subject_train_set(
        config,
        raws,
        events,
        meta["filenames"],
        num_general=5,
        num_dino=17,
        num_supression=0,
        shuffle=True,
    )

    # 3. Preprocessing
    print(f"\n{'='*60}\nPREPROCESSING\n{'='*60}")
    filter_obj = Filter(config, online=False)
    all_epochs = []

    for raw, evts, fname, sid in zip(x_train_raw, y_train_evt, filenames, sub_ids):
        f_raw = raw.copy().apply_function(filter_obj.apply_filter_offline)
        f_raw.drop_channels(config.remove_channels)

        epochs = mne.Epochs(
            f_raw,
            evts,
            event_id=evt_id,
            tmin=0.3,
            tmax=3.0,
            preload=True,
            baseline=None,
        )

        if len(epochs) > 0:
            epochs.metadata = pd.DataFrame(
                {
                    "subject_id": [sid] * len(epochs),
                    "session": [_session_id_from_filename(fname)] * len(epochs),
                    "filename": [fname] * len(epochs),
                }
            )
            all_epochs.append(epochs)

    combined_epochs = mne.concatenate_epochs(all_epochs)
    X, y = combined_epochs.get_data(), combined_epochs.events[:, 2]
    groups = combined_epochs.metadata["session"].values

    # 4. Cross-Validation
    cv_result = run_cv_for_config(combined_epochs, groups, X, y, config, len(evt_id))

    # Display Results
    table = MetricsTable()
    table.add_rows([cv_result])
    table.display()

    # 5. Final Model Training
    print("\nTraining Final Model on full dataset...")
    X_win, y_win, win_groups = epochs_to_windows(
        combined_epochs, groups, config.window_size, config.step_size
    )

    ar_final = ArtefactRemoval()
    ar_final.get_rejection_thresholds(X_win, config)
    X_cl, y_cl, gr_cl = ar_final.reject_bad_epochs_riemann(X_win, y_win, win_groups)

    final_clf = RiemannianClf(cov_est="lwf")
    final_clf.fit_centered(X_cl, y_cl, gr_cl)

    # 6. Save Artifacts
    save_dir = root / "resources" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    final_clf.save(str(save_dir / "model.pkl"))
    with open(save_dir / "ar.pkl", "wb") as f:
        pickle.dump(ar_final, f)

    print(f"Final artifacts saved to: {save_dir}")
    return [cv_result], pd.DataFrame([cv_result])


if __name__ == "__main__":
    run_baseline_comparison_pipeline()

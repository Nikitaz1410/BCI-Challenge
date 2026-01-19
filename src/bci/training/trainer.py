"""Cross-validation and training helpers.

This module runs grouped cross-validation while ensuring window extraction
is performed inside each fold (to prevent leakage). It trains a Riemannian
classifier on covariance matrices computed from windows.

All heavy computation lives here so `main_offline.py` stays minimal.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any

from bci.Training.cv import grouped_kfold_indices, epochs_windows_from_fold
from bci.models.riemann import RiemannianClf, compute_covariances
from bci.evaluation.metrics import compute_ece
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    ece = float(compute_ece(y_true, y_proba, n_bins=10)) if y_proba is not None else float("nan")
    return {"acc": acc, "balanced_acc": bacc, "f1": f1, "ece": ece}


def run_cross_validation(
    epochs,
    labels: np.ndarray,
    groups: np.ndarray,
    config: Dict[str, Any],
    n_splits: int = 5,
    random_state: int = 42,
):
    """Run grouped CV on `epochs`.

    Parameters
    ----------
    epochs : mne.Epochs
        Full epochs (n_epochs, n_channels, n_times)
    labels : np.ndarray
        Labels per epoch (n_epochs,)
    groups : np.ndarray
        Group ids per epoch (n_epochs,) used to prevent leakage
    config : dict
        Config containing window_size and step_size (in samples)
    n_splits : int
        Number of folds

    Returns
    -------
    dict
        Aggregated fold metrics and final test/train results
    """
    results = {
        "folds": [],
    }

    splits = grouped_kfold_indices(groups, n_splits=n_splits)

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        # Extract windows inside fold to avoid leakage
        fold_data = epochs_windows_from_fold(
            epochs, train_idx, val_idx,
            window_size=int(config.get("window_size", 250)),
            step_size=int(config.get("step_size", 32)),
        )

        X_tr, y_tr = fold_data["X_train"], fold_data["y_train"]
        X_val, y_val = fold_data["X_val"], fold_data["y_val"]

        # Compute covariance matrices for Riemannian classifier
        covs_tr = compute_covariances(X_tr)
        covs_val = compute_covariances(X_val)

        # Train classifier
        clf = RiemannianClf()
        clf.fit(covs_tr, y_tr)

        # Predict
        y_pred = clf.predict(covs_val)
        try:
            y_proba = clf.predict_proba(covs_val)
        except Exception:
            y_proba = None

        metrics = _compute_metrics(y_val, y_pred, y_proba)
        results["folds"].append(metrics)

    # Aggregate fold metrics (mean)
    import statistics

    agg = {}
    for key in results["folds"][0].keys():
        agg[key] = float(statistics.mean([f[key] for f in results["folds"]]))

    results["aggregate"] = agg

    # Train final model on all epochs and return it (useful for test evaluation)
    # Extract windows from full training epochs
    X_all, y_all = epochs_windows_from_fold(epochs, np.arange(len(epochs)), np.array([], dtype=int),
                                           window_size=int(config.get("window_size", 250)),
                                           step_size=int(config.get("step_size", 32)),)["X_train"], None
    # Note: epochs_windows_from_fold with empty val_idx returns only X_train; we only need X_all and its labels
    # However for simplicity, recompute labels from epochs
    y_all = epochs.events[:, -1].copy()

    covs_all = compute_covariances(X_all)
    final_clf = RiemannianClf()
    final_clf.fit(covs_all, y_all)

    return {"results": results, "final_model": final_clf}

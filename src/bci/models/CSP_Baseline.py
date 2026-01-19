"""
This module provides a simple wrapper around CSP (Common Spatial Patterns)
feature extraction combined with a user-selectable classifier (SVM, LDA, etc.).

The interface provides the following methods:
- `CSP_Model.fit(...)` trains from numpy arrays.
- `CSP_Model.predict(...)` predicts from numpy arrays.
- `CSP_Model.predict_proba(...)` returns class probabilities from numpy arrays.
- `CSP_Model.save(...)` and `CSP_Model.load(...)` persist and restore models.

Usage:
    model = CSP_Model(classifier="svm").fit(signals, labels)
    preds = model.predict(signals)
    probs = model.predict_proba(signals)
    model.save("/path/to/model.pkl")
    restored = CSP_Model.load("/path/to/model.pkl")
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Set MNE log level to avoid verbose output
mne.set_log_level(verbose='WARNING', return_old_level=False, add_frames=None)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _as_int_labels(labels: Optional[np.ndarray], fallback_size: int) -> np.ndarray:
    """
    Convert labels to a 1D integer array.

    If labels is None, return zeros of length `fallback_size`.
    Accepts either class indices (shape [N]) or one-hot arrays (shape [N, C]).
    """
    if labels is None:
        return np.zeros(fallback_size, dtype=np.int64)

    labels = np.asarray(labels)
    if labels.ndim == 1:
        return labels.astype(np.int64)
    if labels.ndim == 2:
        return labels.argmax(axis=1).astype(np.int64)

    raise ValueError("Labels must be 1D class indices or 2D one-hot arrays.")


def _create_classifier(classifier_type: str) -> Any:
    """
    Create a classifier instance based on the classifier type string.

    Args:
        classifier_type: One of 'svm', 'lda', 'logreg', 'lr', 'csp-svm', 'csp-lda', 'csp-logreg', 'csp-lr'

    Returns:
        A scikit-learn classifier instance.
    """
    classifier_type = str(classifier_type or "lda").lower()

    if classifier_type in ("lda", "csp-lda"):
        return LinearDiscriminantAnalysis()
    elif classifier_type in ("svm", "csp-svm"):
        return SVC(probability=True)
    elif classifier_type in ("logreg", "lr", "csp-logreg", "csp-lr"):
        return LogisticRegression(max_iter=1000)
    else:
        raise ValueError(
            f"Unsupported classifier_type '{classifier_type}'. "
            "Expected one of ['lda', 'svm', 'logreg', 'csp-lda', 'csp-svm', 'csp-logreg']."
        )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class CSP_Model:
    """
    High-level wrapper around CSP feature extraction with a configurable classifier.

    This model always uses CSP (Common Spatial Patterns) for feature extraction
    but allows choosing the classifier during initialization (SVM, LDA, Logistic Regression).

    signals: numpy array shaped [N, C, T] of the EEG signals
    labels: class labels as integers or one-hot array shaped [N] or [N, num_classes]

    Usage:
        model = CSP_Model(classifier="svm", n_components=50).fit(signals, labels)
        preds = model.predict(signals)
        probs = model.predict_proba(signals)
        model.save("/path/to/model.pkl")
        restored = CSP_Model.load("/path/to/model.pkl")
    """

    def __init__(
        self,
        classifier: str = "lda",
        n_components: int = 50,
        csp_reg: Optional[float] = None,
        csp_log: bool = True,
        csp_norm_trace: bool = True,
    ) -> None:
        """
        Initialize the CSP Baseline Model.

        Args:
            classifier: Type of classifier to use. One of 'svm', 'lda', 'logreg'.
                Can also use 'csp-svm', 'csp-lda', 'csp-logreg' (prefix is optional).
            n_components: Number of CSP components to extract (default: 50).
            csp_reg: Regularization parameter for CSP (default: None).
            csp_log: Whether to log-transform CSP features (default: True).
            csp_norm_trace: Whether to normalize CSP features by trace (default: True).
        """
        self.classifier_type = classifier
        self.n_components = n_components
        self.csp_reg = csp_reg
        self.csp_log = csp_log
        self.csp_norm_trace = csp_norm_trace

        self._pipeline: Optional[Pipeline] = None
        self._meta: Dict[str, Any] = {}

    # ------------------------- Training & Prediction ---------------------- #

    def fit(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
    ) -> "CSP_Model":
        """
        Train the CSP baseline model from numpy arrays.

        Args:
            signals: numpy array shaped [N, C, T] of EEG signals.
            labels: class indices or one-hot array shaped [N] or [N, num_classes].

        Returns:
            self for method chaining.
        """
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError("Signals must have shape [N, C, T], got shape {}".format(signals.shape))

        labels = _as_int_labels(labels, fallback_size=signals.shape[0])
        n_classes = int(labels.max()) + 1

        # Store metadata
        self._meta = {
            "n_samples": signals.shape[0],
            "n_channels": signals.shape[1],
            "n_timepoints": signals.shape[2],
            "n_classes": n_classes,
            "classifier_type": self.classifier_type,
            "n_components": self.n_components,
            "csp_reg": self.csp_reg,
            "csp_log": self.csp_log,
            "csp_norm_trace": self.csp_norm_trace,
        }

        # Create CSP feature extractor
        csp = mne.decoding.CSP(
            n_components=self.n_components,
            reg=self.csp_reg,
            log=self.csp_log,
            norm_trace=self.csp_norm_trace,
        )

        # Create classifier
        clf = _create_classifier(self.classifier_type)

        # Create pipeline: CSP -> Classifier
        self._pipeline = Pipeline([("CSP", csp), ("clf", clf)])

        # Fit the pipeline
        self._pipeline.fit(signals, labels)

        return self

    def predict(
        self,
        signals: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class labels for given signals using the trained CSP baseline model.

        Args:
            signals: numpy array shaped [N, C, T] of EEG signals.

        Returns:
            numpy array of shape [N] with predicted class labels.
        """
        if self._pipeline is None:
            raise RuntimeError("Model is not trained. Call `fit` or `load` first.")

        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError("Signals must have shape [N, C, T], got shape {}".format(signals.shape))

        preds = self._pipeline.predict(signals)
        return preds

    def predict_proba(
        self,
        signals: np.ndarray,
    ) -> np.ndarray:
        """
        Return class probabilities for given signals.

        Args:
            signals: numpy array shaped [N, C, T] of EEG signals.

        Returns:
            numpy array of shape [N, num_classes] with class probabilities.
            Raises RuntimeError if the classifier does not support predict_proba.
        """
        if self._pipeline is None:
            raise RuntimeError("Model is not trained. Call `fit` or `load` first.")

        if not hasattr(self._pipeline.named_steps["clf"], "predict_proba"):
            raise RuntimeError(
                f"Classifier {type(self._pipeline.named_steps['clf']).__name__} "
                "does not support predict_proba()."
            )

        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError("Signals must have shape [N, C, T], got shape {}".format(signals.shape))

        probs = self._pipeline.predict_proba(signals)
        return probs

    # ------------------------- Persistence helpers ---------------------- #

    def save(self, path: str) -> str:
        """
        Save the trained model to disk.

        Args:
            path: Path where the model should be saved.

        Returns:
            The path where the model was saved.
        """
        if self._pipeline is None:
            raise RuntimeError("Nothing to save; train or load a model first.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        checkpoint = {
            "pipeline": self._pipeline,
            "meta": self._meta,
            "classifier_type": self.classifier_type,
            "n_components": self.n_components,
            "csp_reg": self.csp_reg,
            "csp_log": self.csp_log,
            "csp_norm_trace": self.csp_norm_trace,
        }

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

        return path

    @classmethod
    def load(cls, path: str) -> "CSP_Model":
        """
        Restore a model previously saved with `save`.

        Args:
            path: Path to the saved model file.

        Returns:
            A restored CSP_Model instance.
        """
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        instance = cls(
            classifier=checkpoint.get("classifier_type", "lda"),
            n_components=checkpoint.get("n_components", 30),
            csp_reg=checkpoint.get("csp_reg", None),
            csp_log=checkpoint.get("csp_log", True),
            csp_norm_trace=checkpoint.get("csp_norm_trace", True),
        )

        instance._pipeline = checkpoint["pipeline"]
        instance._meta = checkpoint.get("meta", {})

        return instance


# Convenience functions for pipeline code that prefers functional access


def train_csp_model(
    signals: np.ndarray,
    labels: np.ndarray,
    classifier: str = "lda",
    n_components: int = 50,
) -> CSP_Model:
    """
    Train and return a CSP_Model instance.

    Args:
        signals: numpy array shaped [N, C, T] of EEG signals.
        labels: class indices or one-hot array shaped [N] or [N, num_classes].
        classifier: Type of classifier to use ('svm', 'lda', 'logreg').
        n_components: Number of CSP components to extract.

    Returns:
        A trained CSP_Model instance.
    """
    return CSP_Model(classifier=classifier, n_components=n_components).fit(
        signals=signals, labels=labels
    )


def predict_csp_model(model: CSP_Model, signals: np.ndarray) -> np.ndarray:
    """
    Run prediction using an existing CSP_Model.

    Args:
        model: A trained CSP_Model instance.
        signals: numpy array shaped [N, C, T] of EEG signals.

    Returns:
        numpy array of shape [N] with predicted class labels.
    """
    return model.predict(signals=signals)


def save_csp_model(model: CSP_Model, path: str) -> str:
    """
    Persist a trained CSP_Model to disk.

    Args:
        model: A trained CSP_Model instance.
        path: Path where the model should be saved.

    Returns:
        The path where the model was saved.
    """
    return model.save(path)


def load_csp_model(path: str) -> CSP_Model:
    """
    Restore a previously saved CSP_Model from disk.

    Args:
        path: Path to the saved model file.

    Returns:
        A restored CSP_Model instance.
    """
    return CSP_Model.load(path)

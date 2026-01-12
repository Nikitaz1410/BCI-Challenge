"""
Unified numpy-friendly interface for the Supervised Autoencoder (SAE).

This module wraps the building blocks that live under `SAE_modules` and
exposes a light-weight API that can be reused by both the offline and the
upcoming online BCI pipelines:

- `SAEModel.fit(...)` trains from numpy arrays.
- `SAEModel.predict(...)` predicts from numpy arrays.
- `SAEModel.predict_proba(...)` returns class probabilities from numpy arrays.
- `SAEModel.save(...)` and `SAEModel.load(...)` persist and restore models.

The heavy lifting (architecture, lightning training loop, datasets) remains
in `SAE_modules`; this file simply standardises how we feed numpy data into
the model and how we serialise it.
"""

from __future__ import annotations

import itertools
import math
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .SAE_modules.denoiser import Denoiser
from .SAE_modules.models import convolution_AE
from .SAE_modules.properties import hyper_params as SAE_HYPER_PARAMS
from .SAE_modules.utils import csp_score

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _as_int_labels(labels: Optional[np.ndarray], fallback_size: int) -> np.ndarray:
    """
    Convert labels to a 1D integer array.

    If labels are None, return zeros of length `fallback_size`.
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


def _one_hot(labels: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    labels = labels.astype(np.int64)
    n_classes = int(n_classes or labels.max() + 1)
    one_hot = np.zeros((labels.size, n_classes), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1.0
    return one_hot


def _conv1d_length(length: int, kernel: int, stride: int, padding: int) -> int:
    return math.floor((length + 2 * padding - kernel) / stride + 1)


def _conv_transpose1d_length(
    length: int, kernel: int, stride: int, padding: int, output_padding: int
) -> int:
    return (length - 1) * stride - 2 * padding + kernel + output_padding


def _predict_adjustments(signal_length: int, filters: Optional[list[int]] = None) -> Dict[str, Any]:
    """
    Brute-force a set of paddings/output-paddings that give us a
    reconstruction length equal to the input length.

    We keep the search space small so this stays fast; if no exact match is
    found we raise to make the caller aware of the mismatch.
    """
    filters = filters or SAE_HYPER_PARAMS.get("cnvl_filters", [8, 16, 32])

    enc_kernels = [25, 10, 5]
    enc_strides = [5, 2, 2]
    dec_kernels = [5, 10, 25]
    dec_strides = [2, 2, 5]

    enc_pad_range = range(0, 6)  # small, but typically enough for EEG lengths
    dec_pad_range = range(0, 6)
    dec_out_ranges = [range(0, s) for s in dec_strides]  # output_padding < stride

    for enc_pads in itertools.product(enc_pad_range, repeat=3):
        l1 = _conv1d_length(signal_length, enc_kernels[0], enc_strides[0], enc_pads[0])
        if l1 <= 0:
            continue
        l2 = _conv1d_length(l1, enc_kernels[1], enc_strides[1], enc_pads[1])
        if l2 <= 0:
            continue
        l3 = _conv1d_length(l2, enc_kernels[2], enc_strides[2], enc_pads[2])
        if l3 <= 0:
            continue

        for dec0_pad, dec0_out in itertools.product(dec_pad_range, dec_out_ranges[0]):
            d1 = _conv_transpose1d_length(l3, dec_kernels[0], dec_strides[0], dec0_pad, dec0_out)
            if d1 <= 0:
                continue
            for dec1_pad, dec1_out in itertools.product(dec_pad_range, dec_out_ranges[1]):
                d2 = _conv_transpose1d_length(d1, dec_kernels[1], dec_strides[1], dec1_pad, dec1_out)
                if d2 <= 0:
                    continue
                for dec2_pad, dec2_out in itertools.product(dec_pad_range, dec_out_ranges[2]):
                    d3 = _conv_transpose1d_length(
                        d2, dec_kernels[2], dec_strides[2], dec2_pad, dec2_out
                    )
                    if d3 == signal_length:
                        latent_sz = filters[-1] * l3
                        return {
                            "encoder_pad": [*enc_pads],
                            "decoder_pad": [dec0_pad, dec0_out, dec1_pad, dec1_out, dec2_pad, dec2_out],
                            "latent_sz": latent_sz,
                        }

    raise ValueError(
        "Could not predict padding/output_padding that preserves signal length. "
        "Please provide `model_adjustments` explicitly."
    )


class _NumpySAEDataset(Dataset):
    """
    Minimal dataset wrapper that mirrors `EEGDataSet_signal_by_day` but accepts
    numpy inputs directly.
    """

    def __init__(
        self,
        signals: np.ndarray,
        task_labels: Optional[np.ndarray] = None,
        day_labels: Optional[np.ndarray] = None,
    ) -> None:
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError("Signals must have shape [N, C, T].")

        self.X = torch.as_tensor(signals, dtype=torch.float32)
        self.n_samples = self.X.shape[0]
        self.n_channels = self.X.shape[1]

        task_idx = _as_int_labels(task_labels, fallback_size=self.n_samples)
        self.n_task_labels = int(task_idx.max()) + 1
        task_1hot = _one_hot(task_idx, self.n_task_labels)

        day_idx = _as_int_labels(day_labels, fallback_size=self.n_samples)
        self.n_days_labels = int(day_idx.max()) + 1
        day_1hot = _one_hot(day_idx, self.n_days_labels)

        self.task_labels = torch.as_tensor(task_1hot, dtype=torch.float32)
        self.day_labels = torch.as_tensor(day_1hot, dtype=torch.float32)

    def __getitem__(self, index: int):
        return self.X[index], self.task_labels[index], self.day_labels[index]

    def __len__(self) -> int:
        return self.n_samples

    def getAllItems(self):
        return self.X.float(), self.task_labels, self.day_labels


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class SAEModel:
    """
    High-level wrapper around the Lightning-based SAE denoiser.

    signals: numpy array shaped [N, C, T] of the EEG signals
    task_labels: MI class labels as integers or one-hot array shaped [N] or [N, num_classes]
    day_labels: optional day/session ids, shaped like task_labels

    Usage:
        model = SAEModel().fit(signals, task_labels, day_labels)
        preds = model.predict(noisy_signals, day_labels)
        probs = model.predict_proba(noisy_signals, day_labels)
        model.save("/path/to/ckpt.pt")
        restored = SAEModel.load("/path/to/ckpt.pt")
    """

    def __init__(
        self,
        mode: str = "supervised",
        model_adjustments: Optional[Dict[str, Any]] = None,
        filters: Optional[list[int]] = None,
        classifier: str = "csp-lda",
    ) -> None:
        self.mode = mode
        self.model_adjustments = model_adjustments
        self.filters = filters or SAE_HYPER_PARAMS.get("cnvl_filters", [8, 16, 32])
        self.classifier = classifier

        self._denoiser: Optional[Denoiser] = None
        self._meta: Dict[str, Any] = {}
        self._csp_clf = None

    # ------------------------- Training & predictence --------------------- #

    def fit(
        self,
        signals: np.ndarray,
        task_labels: np.ndarray,
        day_labels: Optional[np.ndarray] = None,
    ) -> "SAEModel":
        """
        Train the SAE from numpy arrays.

        signals: numpy array shaped [N, C, T]
        task_labels: class indices or one-hot array shaped [N] or [N, num_classes]
        day_labels: optional day/session ids, shaped like task_labels
        """
        dataset = _NumpySAEDataset(signals, task_labels, day_labels)

        if self.model_adjustments is None:
            # Automatically compute model adjustments based on signal length
            signal_length = signals.shape[2]  # Get time dimension
            try:
                self.model_adjustments = _predict_adjustments(signal_length, self.filters)
                print(f"Auto-computed model adjustments for signal length {signal_length}: "
                      f"latent_sz={self.model_adjustments['latent_sz']}")
            except ValueError as e:
                # Fallback to default if auto-computation fails
                print(f"Warning: Could not auto-compute adjustments, using defaults. Error: {e}")
                self.model_adjustments = {
                    "encoder_pad": [1, 1, 1],
                    "decoder_pad": [1, 0, 1, 0, 1, 2],
                    "latent_sz": 160,
                }

        self._meta = {
            "n_channels": dataset.n_channels,
            "n_task_labels": dataset.n_task_labels,
            "n_days_labels": dataset.n_days_labels,
            "filters": self.filters,
            "classifier": self.classifier,
        }

        self._denoiser = Denoiser(self.model_adjustments, self.mode)
        self._denoiser.fit(dataset)

        # Train CSP+SVM on denoised signals for end-to-end classification.
        denoised = self._denoise(signals, day_labels=day_labels)
        _, clf = csp_score(
            np.float64(denoised),
            np.asarray(task_labels),
            cv_N=5,
            classifier=False,
            classifier_type=self.classifier,
        )
        self._csp_clf = clf
        return self

    def predict(
        self,
        signals: np.ndarray,
        day_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run end-to-end classification (denoise + CSP/SVM) using a trained or restored SAE.

        signals: numpy array shaped [N, C, T]
        day_labels: optional day/session ids for consistency; if omitted, zeros are used.
        """
        if self._denoiser is None or self._denoiser.model is None or self._csp_clf is None:
            raise RuntimeError("Model is not trained or loaded. Call `fit` or `load` first.")

        denoised = self._denoise(signals, day_labels=day_labels)
        preds = self._csp_clf.predict(np.float64(denoised))
        return preds

    def predict_proba(
        self,
        signals: np.ndarray,
        day_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return class probabilities for end-to-end classification (denoise + CSP/SVM).

        signals: numpy array shaped [N, C, T]
        day_labels: optional day/session ids for consistency; if omitted, zeros are used.

        Returns:
            numpy array of shape [N, num_classes] with class probabilities.
            Raises RuntimeError if the classifier does not support predict_proba.
        """
        if self._denoiser is None or self._denoiser.model is None or self._csp_clf is None:
            raise RuntimeError("Model is not trained or loaded. Call `fit` or `load` first.")

        if not hasattr(self._csp_clf, "predict_proba"):
            raise RuntimeError(
                f"Classifier {type(self._csp_clf).__name__} does not support predict_proba(). "
                "Please use a classifier that provides probability estimates."
            )

        denoised = self._denoise(signals, day_labels=day_labels)
        probs = self._csp_clf.predict_proba(np.float64(denoised))
        return probs

    # ------------------------- Persistence helpers ---------------------- #

    def save(self, path: str) -> str:
        """
        Save the trained model weights and metadata.
        """
        if self._denoiser is None or self._denoiser.model is None:
            raise RuntimeError("Nothing to save; train or load a model first.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "state_dict": self._denoiser.model.state_dict(),
            "mode": self.mode,
            "adjustments": self.model_adjustments,
            "meta": self._meta,
            "hyper_params": SAE_HYPER_PARAMS,
            "csp_clf": self._csp_clf,
        }
        torch.save(checkpoint, path)
        return path

    @classmethod
    def load(cls, path: str, mode: Optional[str] = None) -> "SAEModel":
        """
        Restore a model previously saved with `save`.
        """
        # Allow sklearn pipeline objects inside the checkpoint (CSP+SVM).
        try:
            from torch.serialization import add_safe_globals
            from sklearn.pipeline import Pipeline

            add_safe_globals([Pipeline])
        except Exception:
            # If safe globals aren't available, fall back and hope default loading works.
            pass

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        adjustments = checkpoint["adjustments"]
        meta = checkpoint["meta"]
        csp_clf = checkpoint.get("csp_clf")
        filters = meta.get("filters", SAE_HYPER_PARAMS.get("cnvl_filters", [8, 16, 32]))
        classifier = meta.get("classifier", "csp-lda")
        mode = mode or checkpoint.get("mode", "supervised")

        instance = cls(
            mode=mode,
            model_adjustments=adjustments,
            filters=filters,
            classifier=classifier,
        )
        instance._meta = meta
        instance._csp_clf = csp_clf

        denoiser = Denoiser(adjustments, mode)
        model = convolution_AE(
            meta["n_channels"],
            meta["n_days_labels"],
            meta["n_task_labels"],
            adjustments,
            SAE_HYPER_PARAMS["ae_lrn_rt"],
            filters_n=filters,
            mode=mode,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(denoiser.device)

        denoiser.model = model
        instance._denoiser = denoiser
        return instance

    # ------------------------- Internal helpers ------------------------ #

    def _denoise(self, signals: np.ndarray, day_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Internal helper to run the denoiser and return denoised signals.
        """
        dataset = _NumpySAEDataset(signals, task_labels=None, day_labels=day_labels)
        noisy_signal, _, _ = dataset.getAllItems()

        self._denoiser.model.eval()
        with torch.no_grad():
            preds = self._denoiser.model(noisy_signal.to(self._denoiser.device))
        return preds.detach().cpu().numpy()


# Convenience functions for pipeline code that prefers functional access


def train_sae(
    signals: np.ndarray,
    task_labels: np.ndarray,
    day_labels: Optional[np.ndarray] = None,
    mode: str = "supervised",
    model_adjustments: Optional[Dict[str, Any]] = None,
) -> SAEModel:
    """
    Train and return an SAEModel instance.
    """
    return SAEModel(mode=mode, model_adjustments=model_adjustments).fit(
        signals=signals, task_labels=task_labels, day_labels=day_labels
    )


def predict_sae(model: SAEModel, signals: np.ndarray, day_labels: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Run predictence using an existing SAEModel.
    """
    return model.predict(signals=signals, day_labels=day_labels)


def save_sae(model: SAEModel, path: str) -> str:
    """
    Persist a trained SAEModel to disk.
    """
    return model.save(path)


def load_sae(path: str, mode: Optional[str] = None) -> SAEModel:
    """
    Restore a previously saved SAEModel from disk.
    """
    return SAEModel.load(path, mode=mode)


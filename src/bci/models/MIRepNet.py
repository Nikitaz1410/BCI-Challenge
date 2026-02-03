"""
Unified numpy-friendly interface for the MIRepNet foundation model.

This module wraps the MIRepNet model from the MIRepNet directory and
exposes a light-weight API that can be reused by both the offline and the
upcoming online BCI pipelines:

- `MIRepNetModel.fit(...)` trains (finetunes) from numpy arrays.
- `MIRepNetModel.predict(...)` predicts from numpy arrays.
- `MIRepNetModel.predict_proba(...)` returns class probabilities from numpy arrays.
- `MIRepNetModel.save(...)` and `MIRepNetModel.load(...)` persist and restore models.

The heavy lifting (architecture, training loop, preprocessing) remains
in the MIRepNet subdirectory; this file simply standardises how we feed numpy data into
the model and how we serialise it.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import fractional_matrix_power
from scipy.spatial.distance import cdist

# Import MIRepNet components
import sys
import importlib.util
from pathlib import Path

# Load modules from MIRepNet subdirectory using importlib
_mirepnet_dir = Path(__file__).parent / "MIRepNet"

# Load mlm module
_mlm_path = _mirepnet_dir / "model" / "mlm.py"
_mlm_spec = importlib.util.spec_from_file_location("mirepnet_mlm", _mlm_path)
_mlm_module = importlib.util.module_from_spec(_mlm_spec)
_mlm_spec.loader.exec_module(_mlm_module)
mlm_mask = _mlm_module.mlm_mask

# Load channel_list module
_channel_list_path = _mirepnet_dir / "utils" / "channel_list.py"
_channel_list_spec = importlib.util.spec_from_file_location("mirepnet_channel_list", _channel_list_path)
_channel_list_module = importlib.util.module_from_spec(_channel_list_spec)
_channel_list_spec.loader.exec_module(_channel_list_module)
use_channels_names = _channel_list_module.use_channels_names
channel_positions = _channel_list_module.channel_positions
all_channels_names = _channel_list_module.all_channels_names

# The model expects exactly 45 channels (hardcoded in PatchEmbedding)
# Match the MIRepNet repo's channel template ordering.
TARGET_CHANNELS = use_channels_names

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


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """
    Encode integer labels into a consecutive 0..(C-1) range.

    This mirrors the LabelEncoder usage in the MIRepNet repo.
    """
    unique = np.unique(labels)
    label_to_index = {int(label): int(i) for i, label in enumerate(unique.tolist())}
    index_to_label = {int(i): int(label) for label, i in label_to_index.items()}
    encoded = np.array([label_to_index[int(label)] for label in labels], dtype=np.int64)
    return encoded, label_to_index, index_to_label


def _select_device(device: str = "auto") -> torch.device:
    """
    Select torch device based on device string.
    - auto: cuda (if available) -> mps (if available) -> cpu
    """
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_missing_channels_diff(x: np.ndarray, target_channels: list, actual_channels: list) -> np.ndarray:
    """
    Pad or interpolate channels to match target channel configuration.
    
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    target_channels : list
        List of target channel names
    actual_channels : list
        List of actual channel names in the data
        
    Returns
    -------
    padded : numpy array
        data of shape (num_samples, num_target_channels, num_time_samples)
    """
    B, C, T = x.shape
    num_target = len(target_channels)
    
    # Get positions for channels that exist in both actual_channels and channel_positions
    channels_with_pos = [ch for ch in actual_channels if ch in channel_positions]
    existing_pos = np.array([channel_positions[ch] for ch in channels_with_pos]) if channels_with_pos else np.array([])
    
    W = np.zeros((num_target, C))
    for i, target_ch in enumerate(target_channels):
        if target_ch in actual_channels:
            src_idx = actual_channels.index(target_ch)
            W[i, src_idx] = 1.0
        else:
            if target_ch in channel_positions and len(existing_pos) > 0:
                pos = channel_positions[target_ch]
                dist = cdist([pos], existing_pos)[0]
                weights = 1 / (dist + 1e-6)
                weights /= weights.sum()
                # Map weights back to actual_channels indices
                # Only assign weights to channels that have positions
                for w, ch_name in zip(weights, channels_with_pos):
                    actual_idx = actual_channels.index(ch_name)
                    W[i, actual_idx] += w
            else:
                # Fallback: use uniform weights if no position info available
                W[i] = 1.0 / C
    
    padded = np.zeros((B, num_target, T))
    for b in range(B):
        padded[b] = W @ x[b]
    
    return padded


def EA(x: np.ndarray, refEA: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean Alignment preprocessing.
    
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    refEA : numpy array, optional
        Reference covariance matrix. If None, computes from input data.

    Returns
    -------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    refEA : numpy array
        Reference covariance matrix used for alignment
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    
    # Compute reference covariance if not provided
    if refEA is None:
        refEA = np.mean(cov, 0)
    
    # Small regularization for numerical stability (avoid singular/near-singular matrices)
    eps = 1e-6
    refEA_reg = refEA + eps * np.eye(refEA.shape[0], dtype=refEA.dtype)
    sqrtRefEA = fractional_matrix_power(refEA_reg, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA, refEA


def _normalize_subject_ids(
    subject_ids: Optional[np.ndarray], n_samples: int
) -> np.ndarray:
    """
    Ensure subject_ids is a 1D array aligned with samples.
    If None, assume all samples belong to one subject.
    """
    if subject_ids is None:
        return np.zeros(n_samples, dtype=np.int64)

    subject_ids = np.asarray(subject_ids)
    if subject_ids.ndim != 1:
        raise ValueError("subject_ids must be a 1D array.")
    if subject_ids.shape[0] != n_samples:
        raise ValueError("subject_ids length must match number of samples.")
    return subject_ids.astype(np.int64)


def _apply_ea_by_subject(
    signals: np.ndarray,
    subject_ids: Optional[np.ndarray],
    ref_by_subject: Optional[Dict[int, np.ndarray]] = None,
) -> tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Apply Euclidean Alignment per subject within a batch of trials.

    Each subject gets its own reference covariance computed from the trials
    belonging to that subject in the current batch.
    """
    subject_ids = _normalize_subject_ids(subject_ids, n_samples=signals.shape[0])
    processed = np.zeros_like(signals, dtype=np.float32)
    updated_refs: Dict[int, np.ndarray] = {} if ref_by_subject is None else dict(ref_by_subject)

    for subject_id in np.unique(subject_ids):
        subject_id_int = int(subject_id)
        idx = np.where(subject_ids == subject_id)[0]
        subject_trials = signals[idx].astype(np.float32)

        if ref_by_subject is not None and subject_id_int in ref_by_subject:
            subject_ea, ref = EA(subject_trials, ref_by_subject[subject_id_int])
        else:
            subject_ea, ref = EA(subject_trials)

        processed[idx] = subject_ea
        updated_refs[subject_id_int] = ref

    return processed, updated_refs


class _NumpyMIRepNetDataset(Dataset):
    """
    Minimal dataset wrapper that accepts numpy inputs directly.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError("Signals must have shape [N, C, T].")

        self.X = torch.as_tensor(signals, dtype=torch.float32)
        self.n_samples = self.X.shape[0]
        self.n_channels = self.X.shape[1]

        if labels is not None:
            labels = _as_int_labels(labels, fallback_size=self.n_samples)
            self.y = torch.as_tensor(labels, dtype=torch.long)
        else:
            self.y = None

    def __getitem__(self, index: int):
        if self.y is not None:
            return self.X[index], self.y[index]
        return self.X[index]

    def __len__(self) -> int:
        return self.n_samples


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class MIRepNetModel:
    """
    High-level wrapper around the MIRepNet foundation model.

    signals: numpy array shaped [N, C, T] of the EEG signals
    task_labels: MI class labels as integers or one-hot array shaped [N] or [N, num_classes]

    Usage:
        model = MIRepNetModel().fit(signals, task_labels)
        preds = model.predict(signals)
        probs = model.predict_proba(signals)
        model.save("/path/to/ckpt.pt")
        restored = MIRepNetModel.load("/path/to/ckpt.pt")
    """

    def __init__(
        self,
        batch_size: int = 32,
        epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 1e-6,
        optimizer: str = "adam",
        scheduler: str = "cosine",
        device: str = "auto",
        actual_channels: Optional[list] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize MIRepNet model wrapper.
        
        Parameters
        ----------
        batch_size : int
            Batch size for training (default: 32)
        epochs : int
            Number of training epochs (default: 10)
        lr : float
            Learning rate (default: 0.001)
        weight_decay : float
            Weight decay for optimizer (default: 1e-6)
        optimizer : str
            Optimizer to use: 'adam' or 'sgd' (default: 'adam')
        scheduler : str
            Learning rate scheduler: 'cosine', 'step', or 'none' (default: 'cosine')
        device : str
            Device to use: 'auto', 'cpu', 'cuda', or 'mps' (default: 'auto')
        actual_channels : list, optional
            List of actual channel names in the data. If None, assumes use_channels_names.
        random_state : int, optional
            Random seed for reproducible training (DataLoader shuffle, model init).
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.pretrain_path = os.path.join(current_dir, "MIRepNet", "weight", "MIRepNet.pth")
        self.batch_size = batch_size
        self._random_state = random_state
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.scheduler_type = scheduler
        self.device_str = device
        self.device = _select_device(device)
        # Store actual_channels for interpolation, but we always target 45 channels
        self.actual_channels = actual_channels
        # Store config channels if provided (will be used in fit/predict)
        if actual_channels:
            self._config_channels = actual_channels
        else:
            self._config_channels = None

        self._model: Optional[nn.Module] = None
        self._meta: Dict[str, Any] = {}
        self._n_classes: Optional[int] = None
        self._ea_ref: Optional[np.ndarray] = None  # Kept for backward compat; not used.
        self._ea_ref_by_subject: Optional[Dict[int, np.ndarray]] = None
        self._label_to_index: Optional[Dict[int, int]] = None
        self._index_to_label: Optional[Dict[int, int]] = None

        # Online EA adaptation state
        self._online_ea_ref: Optional[np.ndarray] = None
        self._online_ea_n_samples: int = 0
        self._online_ea_alpha: float = 0.1  # EMA smoothing factor
        self._online_ea_min_samples: int = 20  # Min samples before EMA starts

    # ------------------------- Online EA Adaptation ----------------------- #

    def init_online_ea(
        self,
        alpha: float = 0.1,
        min_samples: int = 20,
    ) -> None:
        """
        Initialize online EA adaptation from stored training reference.

        Call this before starting online inference to enable continuous
        adaptation of the Euclidean Alignment reference.

        Parameters
        ----------
        alpha : float
            EMA smoothing factor (0 < alpha <= 1). Higher values adapt faster.
            Default: 0.1 (slow adaptation, stable).
        min_samples : int
            Minimum number of samples before EMA adaptation starts.
            Until then, covariances are accumulated and averaged.
            Default: 20.
        """
        self._online_ea_alpha = alpha
        self._online_ea_min_samples = min_samples
        self._online_ea_n_samples = 0

        # Initialize from stored training reference (pipeline assumes same subject)
        if self._ea_ref_by_subject and len(self._ea_ref_by_subject) > 0:
            first_ref = next(iter(self._ea_ref_by_subject.values()))
            self._online_ea_ref = first_ref.copy()
            print(f"Online EA initialized from training reference (shape: {self._online_ea_ref.shape})")
        else:
            self._online_ea_ref = None
            print("Online EA will be computed from incoming data (no training reference)")

    def _apply_ea_online(self, signals: np.ndarray) -> np.ndarray:
        """
        Apply Euclidean Alignment with continuous adaptation.

        Updates the EA reference using exponential moving average (EMA)
        of incoming covariance matrices.

        Parameters
        ----------
        signals : np.ndarray
            EEG signals of shape (N, C, T)

        Returns
        -------
        np.ndarray
            EA-aligned signals of shape (N, C, T)
        """
        n_samples = signals.shape[0]
        signals = signals.astype(np.float32)

        # Compute covariance for each sample in batch
        batch_covs = np.zeros((n_samples, signals.shape[1], signals.shape[1]))
        for i in range(n_samples):
            batch_covs[i] = np.cov(signals[i])

        # Mean covariance of current batch
        batch_mean_cov = np.mean(batch_covs, axis=0)

        # Update online EA reference
        if self._online_ea_ref is None:
            # First batch: initialize reference
            self._online_ea_ref = batch_mean_cov
            self._online_ea_n_samples = n_samples
        elif self._online_ea_n_samples < self._online_ea_min_samples:
            # Accumulation phase: running mean until min_samples
            total = self._online_ea_n_samples + n_samples
            self._online_ea_ref = (
                self._online_ea_ref * self._online_ea_n_samples + batch_mean_cov * n_samples
            ) / total
            self._online_ea_n_samples = total
        else:
            # EMA phase: exponential moving average
            self._online_ea_ref = (
                (1 - self._online_ea_alpha) * self._online_ea_ref
                + self._online_ea_alpha * batch_mean_cov
            )
            self._online_ea_n_samples += n_samples

        # Apply EA using current reference
        aligned, _ = EA(signals, refEA=self._online_ea_ref)
        return aligned.astype(np.float32)

    # ------------------------- Training & prediction --------------------- #

    def fit(
        self,
        signals: np.ndarray,
        task_labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
    ) -> "MIRepNetModel":
        """
        Train (finetune) the MIRepNet model from numpy arrays.

        signals: numpy array shaped [N, C, T]
        task_labels: class indices or one-hot array shaped [N] or [N, num_classes]
        subject_ids: optional array shaped [N] mapping each trial to a subject
        """
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError("Signals must have shape [N, C, T].")

        task_labels = _as_int_labels(task_labels, fallback_size=signals.shape[0])
        task_labels, label_to_index, index_to_label = _encode_labels(task_labels)
        self._label_to_index = label_to_index
        self._index_to_label = index_to_label
        self._n_classes = int(len(label_to_index))
        print(f"Training labels (original -> encoded): {label_to_index}")
        print(f"Model will output probabilities for {self._n_classes} classes (0 to {self._n_classes-1})")

        # Preprocess signals: per-subject EA and channel padding
        print("Applying Euclidean Alignment (per-subject)...")
        signals_processed, ea_refs = _apply_ea_by_subject(
            signals.astype(np.float32), subject_ids
        )
        self._ea_ref_by_subject = ea_refs
        
        # Always pad/interpolate to 45 channels (model requirement)
        # The model architecture expects exactly 45 channels (hardcoded in PatchEmbedding)
        target_num_channels = len(TARGET_CHANNELS)  # 45 channels
        if signals_processed.shape[1] != target_num_channels:
            print(f"Interpolating channels from {signals.shape[1]} to {target_num_channels}...")
            # Infer actual channel names if not provided or if count doesn't match
            if self.actual_channels is None or len(self.actual_channels) != signals.shape[1]:
                # Try to match channels from config or use a subset of target channels
                if self._config_channels and len(self._config_channels) == signals.shape[1]:
                    actual_channels = self._config_channels
                elif signals.shape[1] <= len(TARGET_CHANNELS):
                    # Use first N channels from target template
                    actual_channels = TARGET_CHANNELS[:signals.shape[1]]
                else:
                    # Create generic channel names (fallback)
                    actual_channels = [f"CH{i}" for i in range(signals.shape[1])]
                self.actual_channels = actual_channels
            
            signals_processed = pad_missing_channels_diff(
                signals_processed,
                TARGET_CHANNELS,
                self.actual_channels
            )
            print(f"Channel interpolation complete: {signals.shape[1]} -> {target_num_channels} channels")
        else:
            # Channels already match target, but verify they're in the right order
            if self.actual_channels is None:
                self.actual_channels = TARGET_CHANNELS
            print(f"Channels already match target ({signals.shape[1]} channels).")

        # Set seeds for reproducible training (before model init and DataLoader)
        if self._random_state is not None:
            torch.manual_seed(self._random_state)
            np.random.seed(self._random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._random_state)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        train_generator = (
            torch.Generator().manual_seed(self._random_state)
            if self._random_state is not None
            else None
        )

        # Create dataset and dataloader
        dataset = _NumpyMIRepNetDataset(signals_processed, task_labels)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            generator=train_generator,
        )

        # Initialize model
        print(f"Loading pretrained model from {self.pretrain_path}...")
        self._model = mlm_mask(
            n_classes=self._n_classes,
            pretrainmode=False,
            pretrain=self.pretrain_path,
        ).to(self.device)

        # Set up training components
        criterion = nn.CrossEntropyLoss()

        if self.optimizer_type == "adam":
            optimizer = optim.Adam(
                self._model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "sgd":
            optimizer = optim.SGD(
                self._model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        if self.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs
            )
        elif self.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = None

        # Training loop
        print(f"Training for {self.epochs} epochs...")
        self._model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                _, outputs = self._model(data)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            if scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(train_loader)
            accuracy = correct / total * 100
            current_lr = optimizer.param_groups[0]['lr']
            
            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs}: Loss={epoch_loss:.4f}, Acc={accuracy:.2f}%, LR={current_lr:.6f}")

        # Store metadata
        self._meta = {
            "n_channels": len(use_channels_names),
            "n_classes": self._n_classes,
            "label_to_index": self._label_to_index,
            "index_to_label": self._index_to_label,
        }

        return self

    def predict(
        self,
        signals: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run classification using a trained or restored MIRepNet model.

        signals: numpy array shaped [N, C, T]
        subject_ids: optional array shaped [N] mapping each trial to a subject
        """
        if self._model is None:
            raise RuntimeError("Model is not trained or loaded. Call `fit` or `load` first.")

        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError("Signals must have shape [N, C, T].")

        # Preprocess signals: per-subject EA and channel padding.
        # Use stored EA ref from fit() when available so validation/test data
        # is transformed with training statistics (no data leakage in CV).
        signals_processed, ea_refs = _apply_ea_by_subject(
            signals.astype(np.float32),
            subject_ids,
            ref_by_subject=self._ea_ref_by_subject,
        )
        self._ea_ref_by_subject = ea_refs

        # Always pad/interpolate to 45 channels (model requirement)
        target_num_channels = len(TARGET_CHANNELS)  # 45 channels
        if signals_processed.shape[1] != target_num_channels:
            # Use stored actual_channels or infer
            if self.actual_channels is None or len(self.actual_channels) != signals.shape[1]:
                if self._config_channels and len(self._config_channels) == signals.shape[1]:
                    actual_channels = self._config_channels
                elif signals.shape[1] <= len(TARGET_CHANNELS):
                    actual_channels = TARGET_CHANNELS[:signals.shape[1]]
                else:
                    actual_channels = [f"CH{i}" for i in range(signals.shape[1])]
            else:
                actual_channels = self.actual_channels

            signals_processed = pad_missing_channels_diff(
                signals_processed,
                TARGET_CHANNELS,
                actual_channels
            )
        # If channels already match target, no padding needed

        # Create dataset and dataloader
        dataset = _NumpyMIRepNetDataset(signals_processed, labels=None)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self._model.eval()
        predictions = []
        with torch.no_grad():
            for data in data_loader:
                if isinstance(data, tuple):
                    data = data[0]
                data = data.to(self.device)
                _, outputs = self._model(data)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def predict_proba(
        self,
        signals: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        adapt_ea: bool = False,
    ) -> np.ndarray:
        """
        Return class probabilities for classification.

        Parameters
        ----------
        signals : np.ndarray
            EEG signals shaped [N, C, T]
        subject_ids : np.ndarray, optional
            Array shaped [N] mapping each trial to a subject (for offline use)
        adapt_ea : bool
            If True, use online EA adaptation (call init_online_ea first).
            If False, use static EA reference from training (default).

        Returns
        -------
        numpy array of shape [N, num_classes] with class probabilities.
        """
        if self._model is None:
            raise RuntimeError("Model is not trained or loaded. Call `fit` or `load` first.")

        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError("Signals must have shape [N, C, T].")

        # Apply Euclidean Alignment
        if adapt_ea:
            # Online mode: continuous EA adaptation
            signals_processed = self._apply_ea_online(signals)
        else:
            # Offline mode: static EA reference per subject
            signals_processed, ea_refs = _apply_ea_by_subject(
                signals.astype(np.float32),
                subject_ids,
                ref_by_subject=self._ea_ref_by_subject,
            )
            self._ea_ref_by_subject = ea_refs

        # Always pad/interpolate to 45 channels (model requirement)
        target_num_channels = len(TARGET_CHANNELS)  # 45 channels
        if signals_processed.shape[1] != target_num_channels:
            # Use stored actual_channels or infer
            if self.actual_channels is None or len(self.actual_channels) != signals.shape[1]:
                if self._config_channels and len(self._config_channels) == signals.shape[1]:
                    actual_channels = self._config_channels
                elif signals.shape[1] <= len(TARGET_CHANNELS):
                    actual_channels = TARGET_CHANNELS[:signals.shape[1]]
                else:
                    actual_channels = [f"CH{i}" for i in range(signals.shape[1])]
            else:
                actual_channels = self.actual_channels

            signals_processed = pad_missing_channels_diff(
                signals_processed,
                TARGET_CHANNELS,
                actual_channels
            )
        # If channels already match target, no padding needed

        # Create dataset and dataloader
        dataset = _NumpyMIRepNetDataset(signals_processed, labels=None)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self._model.eval()
        probabilities = []
        with torch.no_grad():
            for data in data_loader:
                if isinstance(data, tuple):
                    data = data[0]
                data = data.to(self.device)
                _, outputs = self._model(data)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                probabilities.append(probs.cpu().numpy())

        return np.concatenate(probabilities, axis=0)

    # ------------------------- Persistence helpers ---------------------- #

    def save(self, path: str) -> str:
        """
        Save the trained model weights and metadata.
        """
        if self._model is None:
            raise RuntimeError("Nothing to save; train or load a model first.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "state_dict": self._model.state_dict(),
            "meta": self._meta,
            "n_classes": self._n_classes,
            "actual_channels": self.actual_channels,
            "ea_ref_by_subject": self._ea_ref_by_subject,
            "label_to_index": self._label_to_index,
            "index_to_label": self._index_to_label,
        }
        torch.save(checkpoint, path)
        return path

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "MIRepNetModel":
        """
        Restore a model previously saved with `save`.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint["meta"]
        n_classes = checkpoint.get("n_classes", meta.get("n_classes", 2))
        actual_channels = checkpoint.get("actual_channels")
        ea_ref_by_subject = checkpoint.get("ea_ref_by_subject")
        label_to_index = checkpoint.get("label_to_index")
        index_to_label = checkpoint.get("index_to_label")

        instance = cls(
            device=device,
            actual_channels=actual_channels,
        )
        instance._meta = meta
        instance._n_classes = n_classes
        instance._ea_ref_by_subject = ea_ref_by_subject
        instance._label_to_index = label_to_index
        instance._index_to_label = index_to_label

        # Initialize model
        model = mlm_mask(
            n_classes=n_classes,
            pretrainmode=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(instance.device)

        instance._model = model
        return instance


# Convenience functions for pipeline code that prefers functional access


def train_mirepnet(
    signals: np.ndarray,
    task_labels: np.ndarray,
    subject_ids: Optional[np.ndarray] = None,
    **kwargs
) -> MIRepNetModel:
    """
    Train and return an MIRepNetModel instance.
    """
    return MIRepNetModel(**kwargs).fit(
        signals=signals, task_labels=task_labels, subject_ids=subject_ids
    )


def predict_mirepnet(
    model: MIRepNetModel,
    signals: np.ndarray,
    subject_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run prediction using an existing MIRepNetModel.
    """
    return model.predict(signals=signals, subject_ids=subject_ids)


def save_mirepnet(model: MIRepNetModel, path: str) -> str:
    """
    Persist a trained MIRepNetModel to disk.
    """
    return model.save(path)


def load_mirepnet(path: str, device: str = "auto") -> MIRepNetModel:
    """
    Restore a previously saved MIRepNetModel from disk.
    """
    return MIRepNetModel.load(path, device=device)

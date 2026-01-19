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
# Use first 45 channels from all_channels_names as the target template
TARGET_CHANNELS = all_channels_names[:45]

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


def EA(x: np.ndarray) -> np.ndarray:
    """
    Euclidean Alignment preprocessing.
    
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    -------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


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
        batch_size: int = 8,
        epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 1e-6,
        optimizer: str = "adam",
        scheduler: str = "cosine",
        device: str = "auto",
        actual_channels: Optional[list] = None,
    ) -> None:
        """
        Initialize MIRepNet model wrapper.
        
        Parameters
        ----------
        batch_size : int
            Batch size for training (default: 8)
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
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.pretrain_path = os.path.join(current_dir, "MIRepNet", "weight", "MIRepNet.pth")
        self.batch_size = batch_size
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

    # ------------------------- Training & prediction --------------------- #

    def fit(
        self,
        signals: np.ndarray,
        task_labels: np.ndarray,
    ) -> "MIRepNetModel":
        """
        Train (finetune) the MIRepNet model from numpy arrays.

        signals: numpy array shaped [N, C, T]
        task_labels: class indices or one-hot array shaped [N] or [N, num_classes]
        """
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError("Signals must have shape [N, C, T].")

        task_labels = _as_int_labels(task_labels, fallback_size=signals.shape[0])
        self._n_classes = int(task_labels.max() + 1)

        # Preprocess signals: EA and channel padding
        print("Applying Euclidean Alignment...")
        signals_processed = EA(signals.astype(np.float32))
        
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

        # Create dataset and dataloader
        dataset = _NumpyMIRepNetDataset(signals_processed, task_labels)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
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
        }

        return self

    def predict(
        self,
        signals: np.ndarray,
    ) -> np.ndarray:
        """
        Run classification using a trained or restored MIRepNet model.

        signals: numpy array shaped [N, C, T]
        """
        if self._model is None:
            raise RuntimeError("Model is not trained or loaded. Call `fit` or `load` first.")

        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError("Signals must have shape [N, C, T].")

        # Preprocess signals: EA and channel padding
        signals_processed = EA(signals.astype(np.float32))
        
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
    ) -> np.ndarray:
        """
        Return class probabilities for classification.

        signals: numpy array shaped [N, C, T]

        Returns
        -------
        numpy array of shape [N, num_classes] with class probabilities.
        """
        if self._model is None:
            raise RuntimeError("Model is not trained or loaded. Call `fit` or `load` first.")

        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError("Signals must have shape [N, C, T].")

        # Preprocess signals: EA and channel padding
        signals_processed = EA(signals.astype(np.float32))
        
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

        instance = cls(
            device=device,
            actual_channels=actual_channels,
        )
        instance._meta = meta
        instance._n_classes = n_classes

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
    **kwargs
) -> MIRepNetModel:
    """
    Train and return an MIRepNetModel instance.
    """
    return MIRepNetModel(**kwargs).fit(
        signals=signals, task_labels=task_labels
    )


def predict_mirepnet(model: MIRepNetModel, signals: np.ndarray) -> np.ndarray:
    """
    Run prediction using an existing MIRepNetModel.
    """
    return model.predict(signals=signals)


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

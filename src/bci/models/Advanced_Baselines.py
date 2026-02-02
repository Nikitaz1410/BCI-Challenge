"""
Advanced deep learning model wrappers for BCI classification.

This file provides modular wrappers for PyTorch-based EEG classification models,
all exposing a unified API:

- fit()
- predict()
- predict_proba()
- save()/load()

Supported models (from MIRepNet/model):
- EEGNet
- FBCNet
- IFNet
- Conformer
- DeepConvNet
- ShallowConvNet
- ADFCNN (Net)
- EDPNet
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy import signal as scipy_signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from bci.utils.bci_config import load_config

# Import all model architectures
from bci.models.MIRepNet.model.EEGNet import EEGNet
from bci.models.MIRepNet.model.FBCNet import FBCNet, FBCNet_2
from bci.models.MIRepNet.model.IFNet import IFNet
from bci.models.MIRepNet.model.Conformer import Conformer
from bci.models.MIRepNet.model.Deep_Shallow_Conv import DeepConvNet as _OrigDeepConvNet
from bci.models.MIRepNet.model.Deep_Shallow_Conv import ShallowConvNet as _OrigShallowConvNet
from bci.models.MIRepNet.model.ADFCNN import ADFCNN as _OrigADFCNN
from bci.models.MIRepNet.model.EDPNet import EDPNet


# ---------------------------------------------------------------------------
# Adaptive Model Wrappers (fix hardcoded layer sizes)
# ---------------------------------------------------------------------------

class _SquareActivation(nn.Module):
    """y = x²"""
    def forward(self, x):
        return x.pow(2)


class _LogActivation(nn.Module):
    """y = log(max(x, eps))"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class DeepConvNet(nn.Module):
    """
    Adaptive DeepConvNet that handles variable input sizes.
    Uses GlobalAvgPool before classifier to support short sequences (e.g. 250 samples).
    """
    def __init__(
        self,
        n_classes: int,
        Chans: int,
        Samples: int,
        dropoutRate: float = 0.5,
        bn_track: bool = True,
        TemporalKernel_Times: int = 1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        
        from collections import OrderedDict
        
        # Use smaller kernels and stride for short sequences (< 500 samples)
        k = min(10, max(5, Samples // 50)) * TemporalKernel_Times
        pool_k, pool_s = (3, 3) if Samples >= 500 else (2, 2)
        
        self.block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 25, kernel_size=(1, k), padding=(0, k // 2))),
            ('conv2', nn.Conv2d(25, 25, kernel_size=(Chans, 1))),
            ('bn', nn.BatchNorm2d(25, track_running_stats=bn_track)),
            ('elu', nn.ELU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(1, pool_k), stride=(1, pool_s))),
            ('dropout', nn.Dropout(dropoutRate)),
        ]))
        
        self.block2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(25, 50, kernel_size=(1, k), padding=(0, k // 2))),
            ('bn', nn.BatchNorm2d(50, track_running_stats=bn_track)),
            ('elu', nn.ELU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(1, pool_k), stride=(1, pool_s))),
            ('dropout', nn.Dropout(dropoutRate)),
        ]))
        
        self.block3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(50, 100, kernel_size=(1, k), padding=(0, k // 2))),
            ('bn', nn.BatchNorm2d(100, track_running_stats=bn_track)),
            ('elu', nn.ELU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(1, pool_k), stride=(1, pool_s))),
            ('dropout', nn.Dropout(dropoutRate)),
        ]))
        
        self.block4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(100, 200, kernel_size=(1, k), padding=(0, k // 2))),
            ('bn', nn.BatchNorm2d(200, track_running_stats=bn_track)),
            ('elu', nn.ELU()),
            ('dropout', nn.Dropout(dropoutRate)),
        ]))
        
        # Global average pooling + linear (avoids dimension collapse for short inputs)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_block = nn.Linear(200, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier_block(x)
        return x


class ShallowConvNet(nn.Module):
    """
    Adaptive ShallowConvNet that computes classifier input size dynamically.
    """
    def __init__(
        self,
        n_classes: int,
        Chans: int,
        Samples: int,
        dropoutRate: float = 0.5,
        bn_track: bool = True,
        TemporalKernel_Times: int = 1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        
        from collections import OrderedDict
        
        # Adjust pooling parameters for smaller inputs
        # Original: avgpool (1, 75) stride (1, 15) - too aggressive for 250 samples
        pool_size = min(75, max(15, Samples // 4))
        pool_stride = min(15, max(5, pool_size // 5))
        
        self.block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 40, kernel_size=(1, 25 * TemporalKernel_Times))),
            ('conv2', nn.Conv2d(40, 40, kernel_size=(Chans, 1))),
            ('bn', nn.BatchNorm2d(40, track_running_stats=bn_track)),
            ('act1', _SquareActivation()),
            ('avgp', nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride))),
            ('act2', _LogActivation()),
            ('drop', nn.Dropout(dropoutRate)),
        ]))
        
        # Use LazyLinear to automatically infer input features
        self.classifier_block = nn.Sequential(
            nn.LazyLinear(out_features=n_classes, bias=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier_block(x)
        return x


class ADFCNNNet(nn.Module):
    """
    Adaptive ADFCNN that handles variable input sizes.
    Uses adaptive pooling to ensure classifier compatibility.
    """
    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        sampling_rate: int,
        F1: int = 8,
        D: int = 1,
        drop_out: float = 0.5,
    ):
        super().__init__()
        
        F2 = F1 * D
        self.num_classes = num_classes
        
        # Spectral convolutions with 'same' padding
        self.spectral_1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, min(125, sampling_rate // 2)), padding='same'),
            nn.BatchNorm2d(F1),
        )
        self.spectral_2 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, min(30, sampling_rate // 8)), padding='same'),
            nn.BatchNorm2d(F1),
        )
        
        # Spatial processing - pathway 1
        # Use AvgPool2d instead of AdaptiveAvgPool2d (MPS has bugs with non-divisible sizes)
        self.spatial_1 = nn.Sequential(
            nn.Conv2d(F2, F2, (num_channels, 1), padding=0, groups=F2, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(drop_out),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), padding='valid'),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 32), stride=(1, 32)),  # Fixed pooling for MPS compatibility
            nn.Dropout(drop_out),
        )
        
        # Spatial processing - pathway 2
        self.spatial_2 = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(num_channels, 1), padding='valid'),
            nn.BatchNorm2d(F2),
            _SquareActivation(),
            nn.AvgPool2d(kernel_size=(1, 32), stride=(1, 32)),  # Fixed pooling for MPS compatibility
            _LogActivation(),
            nn.Dropout(drop_out),
        )
        
        # Attention layers
        self.drop = nn.Dropout(drop_out)
        self.w_q = nn.Linear(F2, F2)
        self.w_k = nn.Linear(F2, F2)
        self.w_v = nn.Linear(F2, F2)
        
        # Adaptive classifier using LazyLinear
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ELU(),
            nn.Dropout(drop_out),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import math
        
        x_1 = self.spectral_1(x)
        x_2 = self.spectral_2(x)
        
        x_filter_1 = self.spatial_1(x_1)
        x_filter_2 = self.spatial_2(x_2)
        
        x_noattention = torch.cat((x_filter_1, x_filter_2), 3)
        B2, C2, H2, W2 = x_noattention.shape
        x_attention = x_noattention.reshape(B2, C2, H2 * W2).permute(0, 2, 1)
        
        B, N, C = x_attention.shape
        
        q = self.w_q(x_attention).permute(0, 2, 1)
        k = self.w_k(x_attention).permute(0, 2, 1)
        v = self.w_v(x_attention).permute(0, 2, 1)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        d_k = q.size(-1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).reshape(B, N, C)
        x_attention = x_attention + self.drop(x)
        x_attention = x_attention.reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)
        x = self.drop(x_attention)
        
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_config_values() -> Tuple[float, int, int]:
    """Load default values from the config file."""
    try:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        config_path = project_root / "resources" / "configs" / "bci_config.yaml"
        
        if config_path.exists():
            config = load_config(config_path)
            return config.fs, config.window_size, config.step_size
    except Exception:
        pass
    
    # Fallback defaults
    return 250.0, 250, 32


def _get_default_sfreq() -> float:
    """Load the default sampling frequency from the config file."""
    return _load_config_values()[0]


def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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


def _apply_filter_bank(
    signals: np.ndarray,
    n_bands: int,
    sfreq: float,
    low_freq: float = 4.0,
    high_freq: float = 40.0,
) -> np.ndarray:
    """
    Apply filter bank decomposition to EEG signals.
    Returns (N, n_bands, C, T) for FBCNet.
    """
    N, C, T = signals.shape
    bands = np.linspace(low_freq, high_freq, n_bands + 1)
    band_width = (high_freq - low_freq) / n_bands
    
    filtered = np.zeros((N, n_bands, C, T), dtype=np.float32)
    for b in range(n_bands):
        low, high = bands[b], bands[b + 1]
        nyq = sfreq / 2
        low_norm, high_norm = max(low / nyq, 0.01), min(high / nyq, 0.99)
        b_coef, a_coef = scipy_signal.butter(4, [low_norm, high_norm], btype="band")
        filtered[:, b, :, :] = scipy_signal.filtfilt(b_coef, a_coef, signals, axis=2)
    return filtered


def _extract_classification_output(
    outputs: Union[torch.Tensor, Tuple],
    model_type: str,
) -> torch.Tensor:
    """
    Extract classification logits from model forward output.
    
    Model return signatures (verified against MIRepNet/model):
    - EEGNet, DeepConvNet, ShallowConvNet, EDPNet, FBCNet_2: single tensor (logits)
    - FBCNet, ADFCNN: single tensor (log-probabilities via LogSoftmax)
    - Conformer, IFNet: single tensor (logits)
    """
    if isinstance(outputs, tuple):
        return outputs[-1]
    return outputs


# Models that output log-probabilities (LogSoftmax) instead of logits
_MODELS_WITH_LOG_PROBS = frozenset({"fbcnet", "adfcnn"})


def _prepare_input_tensor(
    signals: np.ndarray,
    model_type: str,
    device: torch.device,
    n_bands: int = 9,
    radix: int = 2,
    sfreq: float = 250.0,
) -> torch.Tensor:
    """
    Prepare input tensor with correct shape for each model type.
    
    Input signals shape: (N, C, T) where N=samples, C=channels, T=timepoints
    
    Different models expect different input shapes:
    - EEGNet, DeepConvNet, ShallowConvNet, ADFCNN: (N, 1, C, T)
    - Conformer, EDPNet: (N, C, T) - Conformer/EDPNet unsqueeze internally
    - FBCNet: (N, nBands, C, T) - filter bank applied
    - FBCNet_2: same as FBCNet
    - IFNet: (N, C*radix, T) - channels replicated by radix
    """
    x = torch.from_numpy(signals.astype(np.float32)).to(device)
    
    # Models expecting (N, 1, C, T) - add channel dimension
    if model_type in ("eegnet", "deepconvnet", "shallowconvnet", "adfcnn"):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (N, C, T) -> (N, 1, C, T)
    
    # FBCNet expects (N, nBands, C, T) - apply filter bank
    elif model_type in ("fbcnet", "fbcnet_2"):
        sig_np = signals if isinstance(signals, np.ndarray) else signals.cpu().numpy()
        filtered = _apply_filter_bank(sig_np, n_bands=n_bands, sfreq=sfreq)
        x = torch.from_numpy(filtered).float().to(device)
    
    # IFNet expects (N, C*radix, T) - replicate channels by radix
    elif model_type == "ifnet":
        if x.ndim == 3:
            x = x.repeat(1, radix, 1)  # (N, C, T) -> (N, C*radix, T)
    
    # Conformer, EDPNet: (N, C, T) - models unsqueeze/expect as-is
    elif model_type in ("conformer", "edpnet"):
        pass  # Already (N, C, T)
    
    return x


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

ModelType = Literal[
    "eegnet", "fbcnet", "fbcnet_2", "ifnet", "conformer",
    "deepconvnet", "shallowconvnet", "adfcnn", "edpnet"
]


@dataclass
class ModelConfig:
    """Configuration for deep learning models."""
    # Common parameters
    n_classes: int = 3
    n_channels: int = 22
    n_samples: int = 250
    sampling_rate: float = 250.0
    dropout_rate: float = 0.5
    
    # EEGNet specific
    eegnet_kern_length: int = 64
    eegnet_f1: int = 8
    eegnet_d: int = 2
    eegnet_f2: int = 16
    eegnet_norm_rate: float = 0.25
    
    # FBCNet specific (stride_factor must divide n_samples, e.g. 250)
    fbcnet_n_bands: int = 9
    fbcnet_m: int = 32
    fbcnet_stride_factor: int = 5  # 250/5=50
    
    # IFNet specific
    ifnet_out_planes: int = 64
    ifnet_kernel_size: int = 63
    ifnet_radix: int = 2
    ifnet_patch_size: int = 125
    
    # Conformer specific
    conformer_emb_size: int = 40
    conformer_depth: int = 6
    
    # DeepConvNet/ShallowConvNet specific
    conv_temporal_kernel_times: int = 1
    conv_bn_track: bool = True
    
    # EDPNet specific
    edpnet_f1: int = 9
    edpnet_f2: int = 48
    edpnet_time_kernel1: int = 50  # Reduced for smaller inputs
    edpnet_pool_kernels: List[int] = field(default_factory=lambda: [25, 50, 100])  # Smaller kernels
    
def _create_model(
    model_type: ModelType,
    config: ModelConfig,
) -> nn.Module:
    """
    Create a model instance based on the model type and configuration.
    """
    model_type = model_type.lower().strip()
    
    if model_type == "eegnet":
        return EEGNet(
            n_classes=config.n_classes,
            Chans=config.n_channels,
            Samples=config.n_samples,
            kernLenght=config.eegnet_kern_length,
            F1=config.eegnet_f1,
            D=config.eegnet_d,
            F2=config.eegnet_f2,
            dropoutRate=config.dropout_rate,
            norm_rate=config.eegnet_norm_rate,
        )
    
    elif model_type == "fbcnet":
        return FBCNet(
            nChan=config.n_channels,
            nTime=config.n_samples,
            nClass=config.n_classes,
            nBands=config.fbcnet_n_bands,
            m=config.fbcnet_m,
            strideFactor=config.fbcnet_stride_factor,
            doWeightNorm=True,
        )
    
    elif model_type == "fbcnet_2":
        input_shape = (1, config.fbcnet_n_bands, config.n_channels, config.n_samples)
        return FBCNet_2(
            n_classes=config.n_classes,
            input_shape=input_shape,
            m=config.fbcnet_m,
            temporal_stride=config.fbcnet_stride_factor,
        )
    
    elif model_type == "ifnet":
        return IFNet(
            in_planes=config.n_channels,
            out_planes=config.ifnet_out_planes,
            kernel_size=config.ifnet_kernel_size,
            radix=config.ifnet_radix,
            patch_size=config.ifnet_patch_size,
            time_points=config.n_samples,
            num_classes=config.n_classes,
        )
    
    elif model_type == "conformer":
        return Conformer(
            emb_size=config.conformer_emb_size,
            depth=config.conformer_depth,
            n_classes=config.n_classes,
            num_cha=config.n_channels,
        )
    
    elif model_type == "deepconvnet":
        return DeepConvNet(
            n_classes=config.n_classes,
            Chans=config.n_channels,
            Samples=config.n_samples,
            dropoutRate=config.dropout_rate,
            bn_track=config.conv_bn_track,
            TemporalKernel_Times=config.conv_temporal_kernel_times,
        )
    
    elif model_type == "shallowconvnet":
        return ShallowConvNet(
            n_classes=config.n_classes,
            Chans=config.n_channels,
            Samples=config.n_samples,
            dropoutRate=config.dropout_rate,
            bn_track=config.conv_bn_track,
            TemporalKernel_Times=config.conv_temporal_kernel_times,
        )
    
    elif model_type == "adfcnn":
        return ADFCNNNet(
            num_classes=config.n_classes,
            num_channels=config.n_channels,
            sampling_rate=int(config.sampling_rate),
        )
    
    elif model_type == "edpnet":
        return EDPNet(
            chans=config.n_channels,
            samples=config.n_samples,
            num_classes=config.n_classes,
            F1=config.edpnet_f1,
            F2=config.edpnet_f2,
            time_kernel1=config.edpnet_time_kernel1,
            pool_kernels=config.edpnet_pool_kernels,
        )
    
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Expected one of: eegnet, fbcnet, fbcnet_2, ifnet, conformer, "
            f"deepconvnet, shallowconvnet, adfcnn, edpnet"
        )


# ---------------------------------------------------------------------------
# Deep Learning BCI Model Wrapper
# ---------------------------------------------------------------------------

class DeepBCIModel:
    """
    Unified wrapper for deep learning EEG classification models.
    
    Provides a consistent interface (fit, predict, predict_proba, save, load)
    for various PyTorch-based architectures.
    
    Parameters
    ----------
    model_type : str
        Type of model to use. One of:
        - 'eegnet': EEGNet architecture
        - 'fbcnet': Filter Bank Common Spatial Pattern Network
        - 'fbcnet_2': Alternative FBCNet implementation
        - 'ifnet': Inter-Frequency Network
        - 'conformer': Conformer (CNN + Transformer)
        - 'deepconvnet': Deep Convolutional Network
        - 'shallowconvnet': Shallow Convolutional Network
        - 'adfcnn': Adaptive Dual-Flow CNN
        - 'edpnet': Efficient Dual-Prototype Network
    
    n_classes : int
        Number of output classes.
    
    n_channels : int
        Number of EEG channels.
    
    n_samples : int
        Number of time samples per epoch.
    
    sampling_rate : float
        Sampling frequency in Hz.
    
    learning_rate : float
        Learning rate for optimizer.
    
    batch_size : int
        Batch size for training.
    
    n_epochs : int
        Number of training epochs.
    
    device : str, optional
        Device to use ('cuda', 'mps', 'cpu', or 'auto').
    
    model_config : ModelConfig, optional
        Additional model-specific configuration.
    
    verbose : bool
        Whether to print training progress.
    
    Examples
    --------
    >>> model = DeepBCIModel(
    ...     model_type='eegnet',
    ...     n_classes=3,
    ...     n_channels=22,
    ...     n_samples=250,
    ... )
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        model_type: ModelType = "eegnet",
        n_classes: int = 3,
        n_channels: int = 22,
        n_samples: int = 250,
        sampling_rate: float = 250.0,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        n_epochs: int = 100,
        device: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        verbose: bool = True,
        weight_decay: float = 1e-4,
        patience: int = 10,
        class_weights: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.model_type = model_type.lower().strip()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.patience = patience
        self.class_weights = class_weights
        self.random_state = random_state
        
        # Set device
        if device is None or device == "auto":
            self._device = _get_device()
        else:
            self._device = torch.device(device)
        
        # Create model config
        if model_config is not None:
            self._config = model_config
        else:
            self._config = ModelConfig(
                n_classes=n_classes,
                n_channels=n_channels,
                n_samples=n_samples,
                sampling_rate=sampling_rate,
            )
        
        # Ensure config matches constructor params
        self._config.n_classes = n_classes
        self._config.n_channels = n_channels
        self._config.n_samples = n_samples
        self._config.sampling_rate = sampling_rate
        
        # Model state
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self._classes: Optional[np.ndarray] = None
        self._fitted: bool = False
        self._meta: Dict[str, Any] = {}
        self._training_history: List[Dict[str, float]] = []
        
        # Data normalization stats (computed during fit)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
    
    def _build_model(self) -> None:
        """Build and initialize the model."""
        self._model = _create_model(self.model_type, self._config)
        self._model = self._model.to(self._device)
        
        self._optimizer = optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate scheduler - reduce on plateau
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
    
    def _get_loss_function(self, labels: np.ndarray) -> nn.Module:
        """
        Get the appropriate loss function.
        
        FBCNet and ADFCNN output log-probabilities (LogSoftmax), so use NLLLoss.
        All other models output logits, so use CrossEntropyLoss.
        """
        if self.class_weights is not None:
            weights = torch.from_numpy(self.class_weights.astype(np.float32)).to(self._device)
        else:
            unique, counts = np.unique(labels, return_counts=True)
            weights = torch.from_numpy(
                (1.0 / counts / len(unique)).astype(np.float32)
            ).to(self._device)
            weights = weights / weights.sum() * len(unique)
        
        if self.model_type in _MODELS_WITH_LOG_PROBS:
            return nn.NLLLoss(weight=weights)
        return nn.CrossEntropyLoss(weight=weights)
    
    def fit(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        val_signals: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
    ) -> "DeepBCIModel":
        """
        Fit the model to the training data.
        
        Parameters
        ----------
        signals : np.ndarray
            Training signals of shape (N, C, T).
        labels : np.ndarray
            Training labels of shape (N,) or (N, n_classes).
        val_signals : np.ndarray, optional
            Validation signals for early stopping.
        val_labels : np.ndarray, optional
            Validation labels for early stopping.
        
        Returns
        -------
        self : DeepBCIModel
            The fitted model.
        """
        # Set seed for reproducible model init and training
        fit_seed = random_state if random_state is not None else self.random_state
        if fit_seed is not None:
            torch.manual_seed(fit_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fit_seed)
        
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError(f"Signals must have shape [N, C, T], got shape {signals.shape}")
        
        labels = _as_int_labels(labels, fallback_size=signals.shape[0])
        self._classes = np.unique(labels)
        
        # Update config based on actual data shape
        self._config.n_channels = signals.shape[1]
        self._config.n_samples = signals.shape[2]
        self._config.n_classes = len(self._classes)
        self.n_channels = signals.shape[1]
        self.n_samples = signals.shape[2]
        self.n_classes = len(self._classes)
        
        # Build model
        self._build_model()
        
        # Compute normalization statistics (per-channel)
        self._mean = signals.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        self._std = signals.std(axis=(0, 2), keepdims=True) + 1e-8  # (1, C, 1)
        
        # Normalize signals
        signals_normalized = (signals - self._mean) / self._std
        
        # Prepare data
        X = _prepare_input_tensor(
            signals_normalized, self.model_type, self._device,
            n_bands=self._config.fbcnet_n_bands,
            radix=self._config.ifnet_radix,
            sfreq=self._config.sampling_rate,
        )
        y = torch.from_numpy(labels).long().to(self._device)
        
        train_dataset = TensorDataset(X, y)
        train_generator = (
            torch.Generator().manual_seed(fit_seed) if fit_seed is not None else None
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=len(train_dataset) > self.batch_size,
            num_workers=0,  # Avoid multiprocessing non-determinism
            generator=train_generator,
        )
        
        # Validation data
        val_loader = None
        if val_signals is not None and val_labels is not None:
            # Normalize validation data using training stats
            val_signals_normalized = (val_signals - self._mean) / self._std
            X_val = _prepare_input_tensor(
                val_signals_normalized, self.model_type, self._device,
                n_bands=self._config.fbcnet_n_bands,
                radix=self._config.ifnet_radix,
                sfreq=self._config.sampling_rate,
            )
            y_val = torch.from_numpy(_as_int_labels(val_labels, val_signals.shape[0])).long().to(self._device)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
        
        # Loss function
        criterion = self._get_loss_function(labels)
        
        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        self._model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                self._optimizer.zero_grad()
                
                outputs = self._model(batch_X)
                outputs = _extract_classification_output(outputs, self.model_type)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                
                self._optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            epoch_loss /= total
            train_acc = correct / total
            
            # Validation
            val_loss = None
            val_acc = None
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                
                # Step the learning rate scheduler
                self._scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                # Step scheduler based on training loss if no validation data
                self._scheduler.step(epoch_loss)
            
            # Record history
            history_entry = {"epoch": epoch + 1, "train_loss": epoch_loss, "train_acc": train_acc}
            if val_loss is not None:
                history_entry["val_loss"] = val_loss
                history_entry["val_acc"] = val_acc
            self._training_history.append(history_entry)
            
            if self.verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.n_epochs} - Loss: {epoch_loss:.4f}, Acc: {train_acc:.4f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                print(msg)
        
        # Restore best model if early stopping was used
        if best_state is not None:
            self._model.load_state_dict(best_state)
        
        self._meta = {
            "n_samples": int(signals.shape[0]),
            "n_channels": int(signals.shape[1]),
            "n_timepoints": int(signals.shape[2]),
            "n_classes": int(len(self._classes)),
            "model_type": self.model_type,
            "final_train_loss": self._training_history[-1]["train_loss"],
            "final_train_acc": self._training_history[-1]["train_acc"],
        }
        
        self._fitted = True
        return self
    
    def _evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Evaluate model on a data loader."""
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                outputs = self._model(batch_X)
                outputs = _extract_classification_output(outputs, self.model_type)
                
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        self._model.train()
        return total_loss / total, correct / total
    
    def _ensure_fitted(self) -> None:
        """Ensure model is fitted before prediction."""
        if not self._fitted:
            raise RuntimeError("Model is not trained. Call `fit` or `load` first.")
    
    def predict(self, signals: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Parameters
        ----------
        signals : np.ndarray
            Input signals of shape (N, C, T).
        
        Returns
        -------
        predictions : np.ndarray
            Predicted class labels of shape (N,).
        """
        self._ensure_fitted()
        proba = self.predict_proba(signals)
        return proba.argmax(axis=1)
    
    def predict_proba(self, signals: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Parameters
        ----------
        signals : np.ndarray
            Input signals of shape (N, C, T).
        
        Returns
        -------
        probabilities : np.ndarray
            Class probabilities of shape (N, n_classes).
        """
        self._ensure_fitted()
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError(f"Signals must have shape [N, C, T], got shape {signals.shape}")
        
        # Apply normalization using stored stats
        if self._mean is not None and self._std is not None:
            signals = (signals - self._mean) / self._std
        
        self._model.eval()
        X = _prepare_input_tensor(
            signals, self.model_type, self._device,
            n_bands=self._config.fbcnet_n_bands,
            radix=self._config.ifnet_radix,
            sfreq=self._config.sampling_rate,
        )
        
        all_proba = []
        batch_size = self.batch_size
        
        with torch.no_grad():
            for i in range(0, X.size(0), batch_size):
                batch_X = X[i:i + batch_size]
                outputs = self._model(batch_X)
                outputs = _extract_classification_output(outputs, self.model_type)
                
                # FBCNet and ADFCNN output log-probabilities; others output logits
                if self.model_type in _MODELS_WITH_LOG_PROBS:
                    proba = torch.exp(outputs)
                else:
                    proba = torch.softmax(outputs, dim=1)
                all_proba.append(proba.cpu().numpy())
        
        return np.vstack(all_proba)
    
    def save(self, path: str) -> str:
        """
        Save the model to disk.
        
        Parameters
        ----------
        path : str
            Path to save the model.
        
        Returns
        -------
        path : str
            The path where the model was saved.
        """
        self._ensure_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        checkpoint = {
            "model_type": self.model_type,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "config": self._config,
            "classes": self._classes,
            "meta": self._meta,
            "training_history": self._training_history,
            "norm_mean": self._mean,
            "norm_std": self._std,
            "hyperparams": {
                "n_classes": self.n_classes,
                "n_channels": self.n_channels,
                "n_samples": self.n_samples,
                "sampling_rate": self.sampling_rate,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "weight_decay": self.weight_decay,
                "patience": self.patience,
            },
        }
        
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        return path
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DeepBCIModel":
        """
        Load a model from disk.
        
        Parameters
        ----------
        path : str
            Path to the saved model.
        device : str, optional
            Device to load the model to.
        
        Returns
        -------
        model : DeepBCIModel
            The loaded model.
        """
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        
        hyperparams = checkpoint.get("hyperparams", {})
        
        instance = cls(
            model_type=checkpoint["model_type"],
            n_classes=hyperparams.get("n_classes", 3),
            n_channels=hyperparams.get("n_channels", 22),
            n_samples=hyperparams.get("n_samples", 250),
            sampling_rate=hyperparams.get("sampling_rate", 250.0),
            learning_rate=hyperparams.get("learning_rate", 1e-3),
            batch_size=hyperparams.get("batch_size", 32),
            n_epochs=hyperparams.get("n_epochs", 100),
            device=device,
            model_config=checkpoint.get("config"),
            weight_decay=hyperparams.get("weight_decay", 1e-4),
            patience=hyperparams.get("patience", 10),
        )
        
        instance._config = checkpoint.get("config", instance._config)
        instance._classes = checkpoint.get("classes")
        instance._meta = checkpoint.get("meta", {})
        instance._training_history = checkpoint.get("training_history", [])
        instance._mean = checkpoint.get("norm_mean")
        instance._std = checkpoint.get("norm_std")
        
        # Build and load model
        instance._build_model()
        instance._model.load_state_dict(checkpoint["model_state_dict"])
        
        if "optimizer_state_dict" in checkpoint:
            try:
                instance._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass  # Optimizer state may not be compatible
        
        instance._fitted = True
        return instance
    
    def get_training_history(self) -> List[Dict[str, float]]:
        """Get the training history."""
        return self._training_history.copy()
    
    def get_model(self) -> nn.Module:
        """Get the underlying PyTorch model."""
        self._ensure_fitted()
        return self._model


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def create_eegnet(
    n_classes: int = 3,
    n_channels: int = 22,
    n_samples: int = 250,
    **kwargs,
) -> DeepBCIModel:
    """Create an EEGNet model wrapper."""
    return DeepBCIModel(
        model_type="eegnet",
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        **kwargs,
    )


def create_conformer(
    n_classes: int = 3,
    n_channels: int = 22,
    n_samples: int = 250,
    emb_size: int = 40,
    depth: int = 6,
    **kwargs,
) -> DeepBCIModel:
    """Create a Conformer model wrapper."""
    config = ModelConfig(
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        conformer_emb_size=emb_size,
        conformer_depth=depth,
    )
    return DeepBCIModel(
        model_type="conformer",
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        model_config=config,
        **kwargs,
    )


def create_deepconvnet(
    n_classes: int = 3,
    n_channels: int = 22,
    n_samples: int = 250,
    **kwargs,
) -> DeepBCIModel:
    """Create a DeepConvNet model wrapper."""
    return DeepBCIModel(
        model_type="deepconvnet",
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        **kwargs,
    )


def create_shallowconvnet(
    n_classes: int = 3,
    n_channels: int = 22,
    n_samples: int = 250,
    **kwargs,
) -> DeepBCIModel:
    """Create a ShallowConvNet model wrapper."""
    return DeepBCIModel(
        model_type="shallowconvnet",
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        **kwargs,
    )


def create_edpnet(
    n_classes: int = 3,
    n_channels: int = 22,
    n_samples: int = 250,
    **kwargs,
) -> DeepBCIModel:
    """Create an EDPNet model wrapper."""
    return DeepBCIModel(
        model_type="edpnet",
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        **kwargs,
    )


def create_fbcnet(
    n_classes: int = 3,
    n_channels: int = 22,
    n_samples: int = 250,
    n_bands: int = 9,
    **kwargs,
) -> DeepBCIModel:
    """Create an FBCNet model wrapper."""
    config = ModelConfig(
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        fbcnet_n_bands=n_bands,
    )
    return DeepBCIModel(
        model_type="fbcnet",
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        model_config=config,
        **kwargs,
    )


def create_ifnet(
    n_classes: int = 3,
    n_channels: int = 22,
    n_samples: int = 250,
    **kwargs,
) -> DeepBCIModel:
    """Create an IFNet model wrapper."""
    return DeepBCIModel(
        model_type="ifnet",
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        **kwargs,
    )


def create_adfcnn(
    n_classes: int = 3,
    n_channels: int = 22,
    n_samples: int = 250,
    sampling_rate: float = 250.0,
    **kwargs,
) -> DeepBCIModel:
    """Create an ADFCNN model wrapper."""
    return DeepBCIModel(
        model_type="adfcnn",
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        sampling_rate=sampling_rate,
        **kwargs,
    )



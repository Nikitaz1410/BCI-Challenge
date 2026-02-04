"""
Neural network model architectures for EEG classification.

Available models:
- EEGNet: Compact CNN for EEG
- FBCNet: Filter Bank Common Spatial Pattern Network
- IFNet: Inter-Frequency Network
- Conformer: CNN + Transformer hybrid
- DeepConvNet, ShallowConvNet: Classic deep learning for EEG
- ADFCNN (Net): Adaptive Dual-Flow CNN
- mlm_mask: Masked Language Model style network
"""

from .EEGNet import EEGNet
from .FBCNet import FBCNet
from .IFNet import IFNet
from .Conformer import Conformer
from .Deep_Shallow_Conv import DeepConvNet, ShallowConvNet
from .ADFCNN import Net as ADFCNNNet, ADFCNN
from .mlm import mlm_mask

__all__ = [
    "EEGNet",
    "FBCNet",
    "IFNet",
    "Conformer",
    "DeepConvNet",
    "ShallowConvNet",
    "ADFCNN",
    "ADFCNNNet",
    "mlm_mask",
]

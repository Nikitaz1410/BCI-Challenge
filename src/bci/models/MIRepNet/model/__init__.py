"""
Neural network model architectures for EEG classification.

Available models:
- EEGNet: Compact CNN for EEG
- FBCNet, FBCNet_2: Filter Bank Common Spatial Pattern Networks
- IFNet: Inter-Frequency Network
- Conformer: CNN + Transformer hybrid
- DeepConvNet, ShallowConvNet: Classic deep learning for EEG
- ADFCNN (Net): Adaptive Dual-Flow CNN
- EDPNet: Efficient Dual-Prototype Network
- mlm_mask: Masked Language Model style network
"""

from .EEGNet import EEGNet
from .FBCNet import FBCNet, FBCNet_2
from .IFNet import IFNet
from .Conformer import Conformer
from .Deep_Shallow_Conv import DeepConvNet, ShallowConvNet
from .ADFCNN import Net as ADFCNNNet, ADFCNN
from .EDPNet import EDPNet
from .mlm import mlm_mask

__all__ = [
    "EEGNet",
    "FBCNet",
    "FBCNet_2",
    "IFNet",
    "Conformer",
    "DeepConvNet",
    "ShallowConvNet",
    "ADFCNN",
    "ADFCNNNet",
    "EDPNet",
    "mlm_mask",
]

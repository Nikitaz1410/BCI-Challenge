"""MIRepNet model implementations for EEG classification."""

# Re-export MIRepNetModel from the sibling module MIRepNet.py (the package
# directory shadows the file, so "from bci.models.MIRepNet import MIRepNetModel"
# would otherwise not find the class).
from pathlib import Path
import importlib.util

_wrapper_path = Path(__file__).resolve().parent.parent / "MIRepNet.py"
_spec = importlib.util.spec_from_file_location("bci.models._mirepnet_wrapper", _wrapper_path)
_wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wrapper)

MIRepNetModel = _wrapper.MIRepNetModel
EA = _wrapper.EA
pad_missing_channels_diff = _wrapper.pad_missing_channels_diff
TARGET_CHANNELS = _wrapper.TARGET_CHANNELS

__all__ = ["MIRepNetModel", "EA", "pad_missing_channels_diff", "TARGET_CHANNELS"]

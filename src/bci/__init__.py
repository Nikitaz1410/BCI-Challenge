"""
BCI Pipeline Modules
Exports all processing modules
"""
"""
import importlib.util
from pathlib import Path

# Import preprocessing module from 2_preprocessing folder
_preprocessing_path = Path(__file__).parent / "2_preprocessing" / "__init__.py"

if _preprocessing_path.exists():
    spec = importlib.util.spec_from_file_location("preprocessing", _preprocessing_path)
    preprocessing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocessing)
else:
    raise ImportError(f"Could not find preprocessing module at {_preprocessing_path}")

__all__ = ["preprocessing"]

"""
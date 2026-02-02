import re
from typing import Any, Dict, Optional

from bci.models.Baseline import AllRounderBCIModel
from bci.models.MIRepNet import MIRepNetModel
from bci.models.riemann import RiemannianClf
from bci.models.SAE import SAEModel


def choose_model(model_name: str, model_params: Optional[Dict[str, Any]] = None):
    """
    Instantiates and returns the appropriate model class based on the model name
    and a dictionary of parameters.

    Args:
        model_name (str): Name of the model to choose (case-insensitive).
        model_params (dict, optional): Dictionary of arguments to unpack into the
                                       model constructor. Defaults to empty dict.

    Returns:
        The instantiated model object.

    Raises:
        ValueError: If the model name is not recognized.
    """
    # 1. Handle mutable default argument safely
    if model_params is None:
        model_params = {}

    # 2. Define available models mapping (Name -> Class)
    # Note: We map to the Class itself, not an instance of the class
    model_registry = {
        "riemann": RiemannianClf,
        "sae": SAEModel,
        "baseline": AllRounderBCIModel,
        "mirepnet": MIRepNetModel,
    }

    # 3. Normalize input
    key = model_name.lower().strip()

    # 4. Instantiate or Raise Error
    if key in model_registry:
        model_class = model_registry[key]
        return model_class(**model_params)  # Unpacking the dict here
    else:
        valid_options = list(model_registry.keys())
        raise ValueError(
            f"Model '{model_name}' is not recognized. Available options: {valid_options}"
        )


def _session_id_from_filename(filename: str) -> str:
    """
    Extract session identifier from BIDS-style filename for CV grouping.

    E.g. "sub-P999_ses-S002_task-dino_run-001_eeg_raw" -> "sub-P999_ses-S002".
    Multiple files (runs) from the same session get the same ID so they stay
    in the same CV fold. If the pattern is not found, returns the full filename
    (one file = one group).
    """
    # Strip _raw suffix if present
    base = filename.replace("_raw", "").strip("_")
    match = re.match(r"(sub-[^_]+_ses-[^_]+)", base)
    if match:
        return match.group(1)
    return filename

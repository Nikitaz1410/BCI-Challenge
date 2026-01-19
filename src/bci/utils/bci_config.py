import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class EEGConfig:
    # Signal processing parameters
    window_size: int
    step_size: int
    fs: float
    frequencies: List[float]
    order: int

    # EEG Hardware/Channel setup
    channels: List[str]

    # Data and Classification parameters
    model: str
    subjects_ids: List[int]
    n_folds: int
    split: float
    random_state: int
    test: str

    # Online Mode
    online: str
    replay_subject_id: str
    ip: str
    port: int
    classification_threshold: float
    classification_buffer: int

    # Optional with default
    remove_channels: Optional[List[str]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, file_path: str) -> "EEGConfig":
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Filter for expected keys only
        expected_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in expected_keys}

        return cls(**filtered_dict)

    @property
    def subjects(self) -> List[int]:
        """
        Returns the actual list of subjects to process.
        Logic:
          - If len == 2: Interpretation is [Start, End] -> returns range(Start, End + 1)
          - If len != 2: Interpretation is explicit list -> returns list as-is
        """
        if len(self.subjects_ids) == 2:
            return list(range(self.subjects_ids[0], self.subjects_ids[1] + 1))
        return self.subjects_ids


def load_config(file_path: Path) -> EEGConfig:
    """
    Safely loads a YAML file into the EEGConfig class.

    Args:
        file_path (str): Path to the .yaml configuration file.

    Returns:
        EEGConfig: An instance of the configuration class.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML content is not a valid dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")

    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError("The configuration file is empty.")

    if not isinstance(config_dict, dict):
        raise ValueError("The YAML file must resolve to a dictionary/key-value pairs.")

    # Introspect the EEGConfig class to get a set of allowed field names
    # This prevents the code from crashing if your YAML has extra keys (e.g., notes/comments)
    valid_fields = {field.name for field in fields(EEGConfig)}

    # Create a new dictionary containing only the keys that exist in our class
    clean_config = {k: v for k, v in config_dict.items() if k in valid_fields}

    # Unpack the cleaned dictionary into the class
    return EEGConfig(**clean_config)

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import yaml


@dataclass
class BCIConfig:
    """
    Data class representing the application configuration.
    Enforces types and provides structure.
    """

    # EEG Specifics
    channels: List[str]
    marker_durations: List[float]
    frequencies: List[float]
    order: int
    fs: float
    worN: int
    classes: List[int]
    window_size: int
    step_size: int

    # Data Splits
    train_size: float
    val_size: float
    test_size: float
    random_state: int

    def __post_init__(self):
        """
        Perform validation after data is loaded.
        """
        # 1. Type Enforcement for Scalars
        if not isinstance(self.window_size, int):
            raise TypeError(
                f"window_size must be an integer, got {type(self.window_size)}"
            )

        if not isinstance(self.step_size, int):
            raise TypeError(f"step_size must be an integer, got {type(self.step_size)}")

        if not isinstance(self.order, int):
            raise TypeError(f"order must be an integer, got {type(self.order)}")

        if not isinstance(self.worN, int):
            raise TypeError(f"worN must be an integer, got {type(self.worN)}")

        if not isinstance(self.fs, (float, int)):
            raise TypeError(f"fs must be a number, got {type(self.fs)}")

        if not isinstance(self.random_state, int):
            raise TypeError(
                f"random_state must be an integer, got {type(self.random_state)}"
            )

        # Helper to check float fields (Splits)
        float_fields = {
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
        }

        for name, value in float_fields.items():
            if not isinstance(value, (float, int)):
                raise TypeError(f"{name} must be a number (float), got {type(value)}")
            if not (0 <= value <= 1):
                raise ValueError(f"{name} must be between 0 and 1, got {value}")

        # 2. Logic Validation (Sum to 1.0)
        total_split = self.train_size + self.test_size
        # Allow for small floating point errors
        if not (0.99 <= total_split <= 1.01):
            raise ValueError(
                f"Split sizes must sum to approx 1.0. Sum is {total_split}"
            )

        # 3. List Validations
        if not isinstance(self.channels, list) or not all(
            isinstance(c, str) for c in self.channels
        ):
            raise TypeError("channels must be a list of strings")

        if not isinstance(self.marker_durations, list) or not all(
            isinstance(m, (int, float)) for m in self.marker_durations
        ):
            raise TypeError("marker_durations must be a list of numbers")

        if not isinstance(self.frequencies, list) or len(self.frequencies) != 2:
            raise ValueError(
                "frequencies must be a list containing exactly 2 numbers [low, high]"
            )

        if not isinstance(self.classes, list) or not all(
            isinstance(c, int) for c in self.classes
        ):
            raise TypeError("classes must be a list of integers")


def load_config(config_path: Union[str, Path] = "config.yaml") -> BCIConfig:
    """
    Reads the YAML file and returns a validated AppConfig object.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with path.open("r") as f:
        # safe_load is recommended to prevent arbitrary code execution
        raw_config = yaml.safe_load(f)

    if not raw_config:
        raise ValueError("Config file is empty")

    try:
        # Unpack dictionary into the dataclass
        config = BCIConfig(
            channels=raw_config.get("channels"),
            marker_durations=raw_config.get("marker_durations"),
            frequencies=raw_config.get("frequencies"),
            order=raw_config.get("order"),
            fs=raw_config.get("fs"),
            worN=raw_config.get("worN"),
            classes=raw_config.get("classes"),
            window_size=raw_config.get("window_size"),
            step_size=raw_config.get("step_size"),
            train_size=raw_config.get("train_size"),
            val_size=raw_config.get("val_size"),
            test_size=raw_config.get("test_size"),
            random_state=raw_config.get("random_state"),
        )
        return config

    except TypeError as e:
        # This catches missing arguments in the AppConfig constructor
        raise TypeError(f"Configuration schema mismatch: {e}")


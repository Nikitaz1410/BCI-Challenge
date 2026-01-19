import mne
import numpy as np
from autoreject import get_rejection_threshold
from typing import Optional, Any, Tuple


class ArtefactRemoval:
    def __init__(self, rejection_threshold: Optional[float] = None) -> None:
        """
        Initialize the ArtefactRemoval class.

        Parameters:
        rejection_threshold (Optional[float]): Threshold for rejecting epochs. If None, must be computed.
        """
        self.rejection_threshold: Optional[float] = rejection_threshold

    def get_rejection_thresholds(self, epoch_data: np.ndarray, config: Any) -> None:
        """
        Compute and set the rejection threshold for EEG epochs.

        Parameters:
        epoch_data (np.ndarray): Epoch data of shape (n_epochs, n_channels, n_times).
        config (Any): Configuration object with 'channels' (list), 'fs' (float).
        """
        # Validate input types
        if not hasattr(config, "channels") or not hasattr(config, "fs"):
            raise AttributeError(
                "Config object must have 'channels' and 'fs' attributes."
            )

        if not isinstance(epoch_data, np.ndarray):
            raise TypeError("epoch_data must be a numpy ndarray.")

        # Create MNE info structure
        info = mne.create_info(
            ch_names=config.channels,
            ch_types="eeg",
            sfreq=config.fs,
        )

        # Create MNE EpochsArray
        try:
            epochs = mne.EpochsArray(
                epoch_data,
                info=info,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create MNE EpochsArray: {e}")

        # Compute rejection threshold using autoreject
        try:
            threshold_dict = get_rejection_threshold(epochs)
            self.rejection_threshold = threshold_dict["eeg"]
        except Exception as e:
            raise RuntimeError(f"Failed to compute rejection threshold: {e}")

    def reject_bad_epochs(
        self, epochs_data: np.ndarray, epochs_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reject bad epochs based on the computed rejection threshold.

        Parameters:
        epochs_data (np.ndarray): Epoch data of shape (n_epochs, n_channels, n_times).
        epochs_labels (np.ndarray): Labels corresponding to each epoch.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of good epochs and their labels.
        """
        if self.rejection_threshold is None:
            raise ValueError("Rejection thresholds have not been computed yet.")

        if not isinstance(epochs_data, np.ndarray):
            raise TypeError("epochs_data must be a numpy ndarray.")
        if not isinstance(epochs_labels, np.ndarray):
            raise TypeError("epochs_labels must be a numpy ndarray.")
        if epochs_data.shape[0] != epochs_labels.shape[0]:
            raise ValueError("Number of epochs and labels must match.")

        good_epochs = []
        good_labels = []

        # Iterate through each epoch and check if it exceeds the rejection threshold
        for i in range(epochs_data.shape[0]):
            epoch = epochs_data[i]
            if not np.any(np.abs(epoch) > self.rejection_threshold):
                good_epochs.append(epoch)
                good_labels.append(epochs_labels[i])

        print(
            f"Rejected {epochs_data.shape[0] - len(good_epochs)} bad epochs out of {epochs_data.shape[0]} total epochs."
        )

        return np.array(good_epochs), np.array(good_labels)

    def reject_bad_epoches_online(self, epoch: np.ndarray) -> None:
        """
        Placeholder for online epoch rejection.

        Parameters:
        epoch (np.ndarray): Single epoch data.
        """
        pass

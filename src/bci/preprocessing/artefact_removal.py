import mne
import numpy as np

from autoreject import get_rejection_threshold


class ArtefactRemoval:
    def __init__(self, rejection_threshold=None) -> None:
        self.rejection_threshold = rejection_threshold

    def get_rejection_thresholds(self, epoch_data, config):
        info = mne.create_info(
            ch_names=config.channels,
            ch_types="eeg",
            sfreq=config.fs,
        )

        epochs = mne.EpochsArray(
            epoch_data,
            info=info,
        )

        self.rejection_threshold = get_rejection_threshold(epochs)["eeg"]

    def reject_bad_epochs(self, epochs_data, epochs_labels):
        if self.rejection_threshold is None:
            raise ValueError("Rejection thresholds have not been computed yet.")

        # Reject directly using the data without creating the epochs
        good_epochs = []
        good_labels = []

        # TODO: Check this for efficiency and correctness
        for i in range(epochs_data.shape[0]):
            epoch = epochs_data[i]
            if not np.any(np.abs(epoch) > self.rejection_threshold):
                good_epochs.append(epoch)
                good_labels.append(epochs_labels[i])

        print(
            f"Rejected {epochs_data.shape[0] - len(good_epochs)} bad epochs out of {epochs_data.shape[0]} total epochs."
        )

        return np.array(good_epochs), np.array(good_labels)

    def reject_bad_epoches_online(self, epoch):
        pass

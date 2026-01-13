#change: CV helpers for grouped cross-validation
"""
Small module providing grouped cross-validation utilities.
Keep main_offline minimal by delegating CV split creation here.
"""
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold


def grouped_kfold_indices(groups: np.ndarray, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return a list of (train_idx, val_idx) index arrays produced by GroupKFold,
    ensuring that no group is split between train and validation.

    Parameters:
    - groups: array-like of group labels (length = n_samples)
    - n_splits: number of folds

    Returns:
    - List of (train_idx, val_idx) tuples
    """
    groups = np.asarray(groups)
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    # GroupKFold yields indices for arrays (0..n_samples-1)
    for train_idx, val_idx in gkf.split(np.zeros(len(groups)), np.zeros(len(groups)), groups):
        splits.append((train_idx, val_idx))
    return splits


#change: convenience function to get fold generator
def grouped_kfold_generator(groups: np.ndarray, n_splits: int = 5):
    for train_idx, val_idx in grouped_kfold_indices(groups, n_splits=n_splits):
        yield train_idx, val_idx


#change: helper to extract windows inside CV folds to avoid leakage
def epochs_windows_from_fold(
    epochs, train_idx: np.ndarray, val_idx: np.ndarray, window_size: int = 250, step_size: int = 16
) -> dict:
    """Create windowed datasets for a train/val fold from an MNE Epochs object.

    Parameters
    ----------
    epochs : mne.Epochs
        Full Epochs object (n_epochs, n_channels, n_times). The indices refer to rows in this object.
    train_idx : np.ndarray
        Indices of epochs to use for training (relative to the epochs object).
    val_idx : np.ndarray
        Indices of epochs to use for validation.
    window_size : int
        Window length in samples.
    step_size : int
        Step/hop in samples.

    Returns
    -------
    dict
        Keys: 'X_train', 'y_train', 'trial_ids_train', 'X_val', 'y_val', 'trial_ids_val'
        - X_* : np.ndarray shaped (n_windows, n_channels, window_size)
        - y_* : np.ndarray shaped (n_windows,)
        - trial_ids_* : np.ndarray shaped (n_windows,) mapping each window back to epoch index

    Notes
    -----
    - Uses `bci.preprocessing.windows.epochs_to_windows` to perform window extraction.
    - Designed to be called inside a CV loop so windows are created only from training/validation epochs.
    """
    # Local import to avoid import-time dependency issues
    from bci.preprocessing.windows import epochs_to_windows
    import numpy as np

    # Select subsets of epochs using indices
    epochs_train = epochs[train_idx]
    epochs_val = epochs[val_idx]

    # Convert epochs -> windows
    X_train, y_train = epochs_to_windows(epochs_train, window_size=window_size, step_size=step_size)
    X_val, y_val = epochs_to_windows(epochs_val, window_size=window_size, step_size=step_size)

    # Build trial id arrays mapping each window back to original epoch index
    # For train: repeat each epoch index by number of windows extracted from that epoch
    def make_trial_ids(epochs_subset, original_indices):
        data = epochs_subset.get_data()
        trial_ids = []
        for i in range(len(data)):
            eeg = data[i]
            nw = 0
            n_channels, n_samples = eeg.shape
            # compute number of windows
            if n_samples >= window_size:
                nw = int((n_samples - window_size) / step_size) + 1
            trial_ids.extend([original_indices[i]] * nw)
        if len(trial_ids) == 0:
            return np.array([], dtype=int)
        return np.array(trial_ids, dtype=int)

    trial_ids_train = make_trial_ids(epochs_train, np.asarray(train_idx))
    trial_ids_val = make_trial_ids(epochs_val, np.asarray(val_idx))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "trial_ids_train": trial_ids_train,
        "X_val": X_val,
        "y_val": y_val,
        "trial_ids_val": trial_ids_val,
    }

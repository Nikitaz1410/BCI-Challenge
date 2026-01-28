"""Windowing utilities for EEG epochs and trial arrays.

This module centralizes overlapping-window extraction and helpers that
convert MNE `Epochs` into windowed numpy arrays suitable for training.

New utilities:
- extract_overlapping_windows(eeg, window_size, step_size)
- epochs_to_windows(epochs, window_size, step_size)

Keep functions simple and dependency-light (numpy + mne).
"""

from __future__ import annotations

import numpy as np
import mne
from typing import Tuple


def extract_overlapping_windows(
    eeg: np.ndarray, window_size: int = 250, step_size: int = 16
) -> np.ndarray:
    """Extract overlapping time windows from a single trial array.

    Parameters
    ----------
    eeg : np.ndarray
        Array shaped (n_channels, n_samples) representing one trial/epoch.
    window_size : int
        Window length in samples.
    step_size : int
        Shift between consecutive windows in samples.

    Returns
    -------
    windows : np.ndarray
        Array shaped (n_windows, n_channels, window_size)
    """
    n_channels, n_samples = eeg.shape

    window_length_samples = int(window_size)
    window_shift_samples = int(step_size)

    if window_length_samples > n_samples:
        # No windows can be extracted; return empty array with correct dims
        return np.zeros((0, n_channels, window_length_samples))

    nwindows = int((n_samples - window_length_samples) / (window_shift_samples)) + 1

    window_starts = np.arange(
        0, n_samples - window_length_samples + 1, window_shift_samples
    ).astype(int)
    window_ends = window_starts + window_length_samples

    windows = np.zeros((nwindows, n_channels, window_length_samples))

    # Extract the windows
    for window_id in range(nwindows):
        start = window_starts[window_id]
        end = window_ends[window_id]

        window = eeg[:, start:end]
        windows[window_id, :, :] = window

    return windows


def epochs_to_windows(
    epochs: mne.Epochs, groups: np.ndarray, window_size: int = 250, step_size: int = 16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert MNE `Epochs` into overlapping windows and labels.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object (n_epochs, n_channels, n_times)
    window_size : int
        Window length in samples.
    step_size : int
        Step length in samples.

    Returns
    -------
    windows : np.ndarray
        Array shaped (n_total_windows, n_channels, window_size)
    labels : np.ndarray
        Integer labels per window (derived from epochs.events[:, -1])
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1].copy()

    windowed_epochs = []
    windowed_labels = []
    windowed_groups = []

    for i, eeg in enumerate(data):
        windows = extract_overlapping_windows(
            eeg, window_size=window_size, step_size=step_size
        )
        if windows.shape[0] == 0:
            continue
        windowed_epochs.append(windows)
        windowed_labels.extend([labels[i]] * windows.shape[0])
        windowed_groups.extend([groups[i]] * windows.shape[0])

    if len(windowed_epochs) == 0:
        return np.zeros((0, data.shape[1], window_size)), np.array([], dtype=int)

    windowed_epochs = np.concatenate(windowed_epochs, axis=0)
    windowed_labels = np.array(windowed_labels)
    windowed_groups = np.array(windowed_groups)

    return windowed_epochs, windowed_labels, windowed_groups


def epochs_windows_from_fold(
    epochs,
    groups,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    window_size: int = 250,
    step_size: int = 16,
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

    # Select subsets of epochs using indices
    epochs_train = epochs[train_idx]
    epochs_val = epochs[val_idx]

    # Convert epochs -> windows
    X_train, y_train, _ = epochs_to_windows(
        epochs_train, groups, window_size=window_size, step_size=step_size
    )
    X_val, y_val, _ = epochs_to_windows(
        epochs_val, groups, window_size=window_size, step_size=step_size
    )

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

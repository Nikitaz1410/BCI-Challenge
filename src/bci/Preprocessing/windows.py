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


def extract_overlapping_windows(eeg: np.ndarray, window_size: int = 250, step_size: int = 16) -> np.ndarray:
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


def epochs_to_windows(epochs: mne.Epochs, window_size: int = 250, step_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
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

    for i, eeg in enumerate(data):
        windows = extract_overlapping_windows(eeg, window_size=window_size, step_size=step_size)
        if windows.shape[0] == 0:
            continue
        windowed_epochs.append(windows)
        windowed_labels.extend([labels[i]] * windows.shape[0])

    if len(windowed_epochs) == 0:
        return np.zeros((0, data.shape[1], window_size)), np.array([], dtype=int)

    windowed_epochs = np.concatenate(windowed_epochs, axis=0)
    windowed_labels = np.array(windowed_labels)

    return windowed_epochs, windowed_labels

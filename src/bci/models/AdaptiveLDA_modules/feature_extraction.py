"""
Feature extraction module for Adaptive LDA.

Extracts Power Spectral Density (PSD) features from EEG signals
following the methodology from Wu et al. (2024).
"""

import numpy as np
from scipy import signal


def extract_psd_features(signals, sfreq=160, freq_range=(8, 30), freq_step=2):
    """
    Extract Power Spectral Density (PSD) features from EEG signals.

    For Motor Imagery: Uses Mu (8-13 Hz) and Beta (13-30 Hz) bands.
    Paper used (1-70 Hz) for speech imagery, adapted here for MI.

    Parameters:
    -----------
    signals : np.ndarray
        Shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    sfreq : float
        Sampling frequency of the EEG data in Hz
    freq_range : tuple
        Frequency range for PSD computation (min_freq, max_freq) in Hz
        Default (8, 30) for motor imagery mu and beta bands
    freq_step : int
        Frequency step for PSD computation in Hz

    Returns:
    --------
    features : np.ndarray
        Shape (n_trials, n_features) or (n_features,)
        PSD features for each trial and channel
    """
    # Handle single trial input
    if signals.ndim == 2:
        signals = signals[np.newaxis, ...]
        single_trial = True
    else:
        single_trial = False

    n_trials, n_channels, n_samples = signals.shape

    # Create frequency bins
    freqs = np.arange(freq_range[0], freq_range[1], freq_step)
    n_freqs = len(freqs)

    # Initialize feature array
    psd_features = np.zeros((n_trials, n_channels * n_freqs))

    # Compute PSD for each trial and channel
    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            # Compute PSD using Welch method
            # Paper uses: 500ms window with 20ms overlap
            nperseg = int(0.5 * sfreq)  # 500ms window
            noverlap = int(0.48 * sfreq)  # 20ms step (480ms overlap)

            f, psd = signal.welch(
                signals[trial_idx, ch_idx, :],
                fs=sfreq,
                nperseg=nperseg,
                noverlap=noverlap
            )

            # Extract PSD at desired frequencies
            for freq_idx, target_freq in enumerate(freqs):
                # Find closest frequency bin
                freq_bin = np.argmin(np.abs(f - target_freq))
                feature_idx = ch_idx * n_freqs + freq_idx
                psd_features[trial_idx, feature_idx] = psd[freq_bin]

    # Return single trial if input was single trial
    if single_trial:
        return psd_features[0]

    return psd_features

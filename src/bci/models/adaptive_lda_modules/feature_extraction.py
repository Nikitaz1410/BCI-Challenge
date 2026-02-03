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


def extract_log_bandpower_features(signals, sfreq=160, mu_band=(8, 12), beta_band=(13, 30), eps=1e-10):
    """
    Extract log bandpower features from EEG signals.
    
    Computes log bandpower for mu and beta frequency bands using Welch's method.
    This is commonly used for motor imagery BCI tasks.
    
    Parameters:
    -----------
    signals : np.ndarray
        Shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    sfreq : float, default=160
        Sampling frequency of the EEG data in Hz
    mu_band : tuple, default=(8, 12)
        Mu band frequency range (low, high) in Hz
    beta_band : tuple, default=(13, 30)
        Beta band frequency range (low, high) in Hz
    eps : float, default=1e-10
        Small epsilon value to avoid log(0)
        
    Returns:
    --------
    features : np.ndarray
        Shape (n_trials, n_channels * n_bands) or (n_channels * n_bands,)
        Log bandpower features for each trial, channel, and band
        Order: [ch0_mu, ch0_beta, ch1_mu, ch1_beta, ...]
    """
    # Handle single trial input
    if signals.ndim == 2:
        signals = signals[np.newaxis, ...]
        single_trial = True
    else:
        single_trial = False
    
    n_trials, n_channels, n_samples = signals.shape
    n_bands = 2  # mu and beta
    
    # Initialize feature array
    features = np.zeros((n_trials, n_channels * n_bands))
    
    # Compute log bandpower for each trial and channel
    for trial_idx in range(n_trials):
        feat_idx = 0
        for ch_idx in range(n_channels):
            # Use Welch's method to compute PSD
            # Default window size: 0.5 seconds, overlap: 0.48 seconds (20ms step)
            nperseg = min(int(0.5 * sfreq), n_samples)
            noverlap = max(0, min(int(0.48 * sfreq), n_samples - int(0.02 * sfreq)))
            
            freqs, psd = signal.welch(
                signals[trial_idx, ch_idx, :],
                fs=sfreq,
                nperseg=nperseg,
                noverlap=noverlap
            )
            
            # Extract mu band power
            mu_mask = (freqs >= mu_band[0]) & (freqs <= mu_band[1])
            mu_power = np.mean(psd[mu_mask]) if mu_mask.any() else eps
            features[trial_idx, feat_idx] = np.log(mu_power + eps)
            feat_idx += 1
            
            # Extract beta band power
            beta_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
            beta_power = np.mean(psd[beta_mask]) if beta_mask.any() else eps
            features[trial_idx, feat_idx] = np.log(beta_power + eps)
            feat_idx += 1
    
    # Return single trial if input was single trial
    if single_trial:
        return features[0]
    
    return features

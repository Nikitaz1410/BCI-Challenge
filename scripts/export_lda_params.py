#!/usr/bin/env python3
"""
Offline LDA Parameter Export Script

This script computes and saves the parameters required by the online Adaptive LDA
implementation from PhysioNet-style EEG motor imagery data.

Output: resources/models/lda_params.pkl containing:
    - "mu": class means, shape (K, D)
    - "sigma_inv": inverse pooled covariance, shape (D, D)
    - "priors": class priors, shape (K,)

Usage:
    python scripts/export_lda_params.py

    # Or with custom subjects:
    python scripts/export_lda_params.py --subjects 1 2 3 4 5

"""

import sys
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy import signal
import mne

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from bci.loading.loading import load_physionet_data
from bci.utils.bci_config import load_config


# ============================================================================
# PREPROCESSING (inline to avoid import issues)
# ============================================================================

def preprocess_raw_simple(raw, config, verbose=False):
    """
    Simple preprocessing: channel selection, bad channel removal,
    average reference, and bandpass filtering.
    """
    raw = raw.copy()

    # Get config values
    channels = getattr(config, 'channels', [])
    remove_channels = getattr(config, 'remove_channels', [])
    frequencies = getattr(config, 'frequencies', [8.0, 30.0])

    # 1. Select channels
    if channels:
        available = raw.ch_names
        to_pick = [ch for ch in channels if ch in available]
        if to_pick:
            raw.pick(to_pick)

    # 2. Remove bad channels
    if remove_channels:
        current = raw.ch_names
        to_drop = [ch for ch in remove_channels if ch in current]
        if to_drop:
            raw.drop_channels(to_drop)

    # 3. Average reference
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

    # 4. Bandpass filter
    raw.filter(
        l_freq=frequencies[0],
        h_freq=frequencies[1],
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose=False
    )

    if verbose:
        print(f"    Preprocessed: {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")

    return raw


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_log_bandpower_features(
    signals: np.ndarray,
    sfreq: float = 160.0,
    mu_band: tuple = (8, 12),
    beta_band: tuple = (13, 30)
) -> np.ndarray:
    """
    Extract log-bandpower features from mu (8-12 Hz) and beta (13-30 Hz) bands.

    Parameters
    ----------
    signals : np.ndarray
        Shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz
    mu_band : tuple
        Mu band frequency range (low, high) in Hz
    beta_band : tuple
        Beta band frequency range (low, high) in Hz

    Returns
    -------
    features : np.ndarray
        Shape (n_trials, n_channels * 2) - [mu_ch1, beta_ch1, mu_ch2, beta_ch2, ...]
    """
    # Handle single trial
    if signals.ndim == 2:
        signals = signals[np.newaxis, ...]
        single_trial = True
    else:
        single_trial = False

    n_trials, n_channels, n_samples = signals.shape
    n_bands = 2  # mu and beta

    features = np.zeros((n_trials, n_channels * n_bands))

    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            # Welch PSD: 500ms window, 480ms overlap
            nperseg = min(int(0.5 * sfreq), n_samples)
            noverlap = min(int(0.48 * sfreq), nperseg - 1)

            f, psd = signal.welch(
                signals[trial_idx, ch_idx, :],
                fs=sfreq,
                nperseg=nperseg,
                noverlap=noverlap
            )

            # Mu band power (8-12 Hz)
            mu_mask = (f >= mu_band[0]) & (f <= mu_band[1])
            if np.any(mu_mask):
                mu_power = np.trapz(psd[mu_mask], f[mu_mask])
            else:
                mu_power = 1e-10
            mu_log = np.log(mu_power + 1e-10)

            # Beta band power (13-30 Hz)
            beta_mask = (f >= beta_band[0]) & (f <= beta_band[1])
            if np.any(beta_mask):
                beta_power = np.trapz(psd[beta_mask], f[beta_mask])
            else:
                beta_power = 1e-10
            beta_log = np.log(beta_power + 1e-10)

            # Store: [mu_ch1, beta_ch1, mu_ch2, beta_ch2, ...]
            features[trial_idx, ch_idx * n_bands + 0] = mu_log
            features[trial_idx, ch_idx * n_bands + 1] = beta_log

    return features[0] if single_trial else features


# ============================================================================
# LDA PARAMETER COMPUTATION (NumPy only)
# ============================================================================

def compute_lda_params(
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6
) -> dict:
    """
    Compute LDA parameters: class means, pooled covariance, and priors.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (N, D)
    y : np.ndarray
        Labels, shape (N,) with values in 0..K-1
    eps : float
        Regularization for covariance matrix (added to diagonal)

    Returns
    -------
    params : dict
        {
            "mu": class means, shape (K, D),
            "sigma_inv": inverse pooled covariance, shape (D, D),
            "priors": class priors, shape (K,)
        }
    """
    N, D = X.shape
    classes = np.unique(y)
    K = len(classes)

    print(f"\n{'='*60}")
    print("COMPUTING LDA PARAMETERS")
    print(f"{'='*60}")
    print(f"  Samples (N):    {N}")
    print(f"  Features (D):   {D}")
    print(f"  Classes (K):    {K}")
    print(f"  Unique labels:  {classes.tolist()}")

    # Remap labels to 0..K-1 if needed
    if not np.array_equal(classes, np.arange(K)):
        print(f"\n  [!] Remapping labels to 0..{K-1}")
        label_map = {old: new for new, old in enumerate(classes)}
        y = np.array([label_map[label] for label in y])
        classes = np.arange(K)

    # 1. Compute class means: mu[k] = mean(X[y == k])
    mu = np.zeros((K, D))
    class_counts = np.zeros(K)

    print(f"\n  Class statistics:")
    for k in range(K):
        mask = (y == k)
        class_counts[k] = np.sum(mask)
        if class_counts[k] == 0:
            raise ValueError(f"Class {k} has no samples!")
        mu[k] = np.mean(X[mask], axis=0)
        print(f"    Class {k}: {int(class_counts[k])} samples")

    # 2. Compute pooled within-class covariance (shared covariance LDA)
    # Sigma = (1 / (N - K)) * sum_k sum_{x in class k} (x - mu_k)(x - mu_k)^T
    Sigma = np.zeros((D, D))

    for k in range(K):
        mask = (y == k)
        X_k = X[mask]  # Samples of class k
        X_centered = X_k - mu[k]  # Center around class mean
        Sigma += X_centered.T @ X_centered  # Outer product sum

    # Normalize by (N - K) for unbiased estimate
    denom = N - K
    if denom <= 0:
        denom = N  # Fallback if not enough samples
        warnings.warn(f"N - K = {N - K} <= 0, using N = {N} for normalization")
    Sigma /= denom

    # 3. Regularize covariance: Sigma_reg = Sigma + eps * I
    print(f"\n  Regularizing covariance (eps={eps})")
    Sigma_reg = Sigma + eps * np.eye(D)

    # Check condition number
    cond_num = np.linalg.cond(Sigma_reg)
    print(f"  Condition number: {cond_num:.2e}")

    if cond_num > 1e6:
        warnings.warn(
            f"Covariance matrix is ill-conditioned (cond={cond_num:.2e}). "
            "Consider increasing eps or reducing features."
        )

    # 4. Compute inverse covariance
    try:
        sigma_inv = np.linalg.inv(Sigma_reg)
        print(f"  Inverse computed successfully")
    except np.linalg.LinAlgError:
        warnings.warn("Standard inversion failed, using pseudo-inverse")
        sigma_inv = np.linalg.pinv(Sigma_reg)

    # Ensure symmetry
    sigma_inv = 0.5 * (sigma_inv + sigma_inv.T)

    # 5. Compute class priors from counts
    priors = class_counts / N
    print(f"\n  Class priors: {priors}")

    # Sanity checks
    assert mu.shape == (K, D), f"mu shape mismatch: {mu.shape} vs ({K}, {D})"
    assert sigma_inv.shape == (D, D), f"sigma_inv shape mismatch: {sigma_inv.shape}"
    assert priors.shape == (K,), f"priors shape mismatch: {priors.shape}"
    assert np.allclose(priors.sum(), 1.0), f"priors don't sum to 1: {priors.sum()}"

    if not np.all(np.isfinite(mu)):
        raise ValueError("mu contains non-finite values")
    if not np.all(np.isfinite(sigma_inv)):
        raise ValueError("sigma_inv contains non-finite values")

    print(f"\n{'='*60}")
    print("LDA PARAMETERS COMPUTED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"  mu shape:        {mu.shape}")
    print(f"  sigma_inv shape: {sigma_inv.shape}")
    print(f"  priors shape:    {priors.shape}")

    return {
        "mu": mu,
        "sigma_inv": sigma_inv,
        "priors": priors
    }


# ============================================================================
# EPOCH EXTRACTION
# ============================================================================

def extract_epochs_from_raw(
    raw,
    events: np.ndarray,
    event_id: dict,
    tmin: float = 0.0,
    tmax: float = 4.0,
    baseline: tuple = None
) -> tuple:
    """
    Extract epochs from raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    events : np.ndarray
        Events array from MNE
    event_id : dict
        Event ID mapping
    tmin, tmax : float
        Epoch time window in seconds
    baseline : tuple or None
        Baseline correction window

    Returns
    -------
    X : np.ndarray
        Epoch data, shape (n_epochs, n_channels, n_samples)
    y : np.ndarray
        Labels, shape (n_epochs,)
    """
    import mne

    epochs = mne.Epochs(
        raw, events, event_id,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False
    )

    X = epochs.get_data()  # (n_epochs, n_channels, n_samples)
    y = epochs.events[:, 2]  # Event codes

    return X, y


# ============================================================================
# MAIN EXPORT FUNCTION
# ============================================================================

def export_lda_params(
    subjects: list = None,
    output_path: str = None,
    config_path: str = None,
    eps: float = 1e-6,
    tmin: float = 0.5,
    tmax: float = 3.5,
    verbose: bool = True
):
    """
    Main function to export LDA parameters from PhysioNet data.

    Parameters
    ----------
    subjects : list
        List of subject IDs to use (default: [1, 2, 3, 4, 5])
    output_path : str
        Output path for lda_params.pkl
    config_path : str
        Path to bci_config.yaml
    eps : float
        Regularization parameter
    tmin, tmax : float
        Epoch time window
    verbose : bool
        Print progress
    """
    import mne
    mne.set_log_level("WARNING")

    # Defaults
    if subjects is None:
        subjects = list(range(1, 16))  # Subjects 1-15

    if output_path is None:
        output_path = ROOT_DIR / "resources" / "models" / "lda_params.pkl"
    else:
        output_path = Path(output_path)

    if config_path is None:
        config_path = ROOT_DIR / "resources" / "configs" / "bci_config.yaml"
    else:
        config_path = Path(config_path)

    print(f"\n{'='*60}")
    print("OFFLINE LDA PARAMETER EXPORT")
    print(f"{'='*60}")
    print(f"  Subjects:    {subjects}")
    print(f"  Config:      {config_path}")
    print(f"  Output:      {output_path}")
    print(f"  Epoch:       [{tmin}, {tmax}] s")
    print(f"  Reg (eps):   {eps}")

    # Load config
    config = load_config(config_path)
    print(f"\n  Channels:    {len(config.channels)} total, {len(config.remove_channels)} removed")
    print(f"  Frequencies: {config.frequencies} Hz")
    print(f"  Fs:          {config.fs} Hz")

    # Load PhysioNet data
    print(f"\n{'='*60}")
    print("LOADING PHYSIONET DATA")
    print(f"{'='*60}")

    raws, events_list, event_id, sub_ids, _ = load_physionet_data(
        subjects=subjects,
        root=ROOT_DIR,
        channels=config.channels
    )

    print(f"\n  Loaded {len(raws)} subjects")
    print(f"  Event ID: {event_id}")

    # Process and extract features
    print(f"\n{'='*60}")
    print("PREPROCESSING AND FEATURE EXTRACTION")
    print(f"{'='*60}")

    all_features = []
    all_labels = []

    for i, (raw, events, sub_id) in enumerate(zip(raws, events_list, sub_ids)):
        print(f"\n  Subject {sub_id} ({i+1}/{len(raws)}):")

        # Preprocess (filter, reference, channel selection)
        raw_clean = preprocess_raw_simple(raw, config, verbose=False)

        # Extract epochs
        X, y = extract_epochs_from_raw(
            raw_clean, events, event_id,
            tmin=tmin, tmax=tmax, baseline=None
        )

        print(f"    Epochs: {X.shape[0]}, Channels: {X.shape[1]}, Samples: {X.shape[2]}")

        # Extract features
        features = extract_log_bandpower_features(
            X, sfreq=config.fs,
            mu_band=(8, 12), beta_band=(13, 30)
        )

        print(f"    Features shape: {features.shape}")
        print(f"    Labels: {np.bincount(y)}")

        all_features.append(features)
        all_labels.append(y)

    # Concatenate all data
    X_all = np.vstack(all_features)
    y_all = np.concatenate(all_labels)

    print(f"\n{'='*60}")
    print("COMBINED DATASET")
    print(f"{'='*60}")
    print(f"  Total samples: {X_all.shape[0]}")
    print(f"  Feature dim:   {X_all.shape[1]}")
    print(f"  Label counts:  {dict(zip(*np.unique(y_all, return_counts=True)))}")

    # Compute LDA parameters
    lda_params = compute_lda_params(X_all, y_all, eps=eps)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(lda_params, f)

    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Saved to: {output_path}")

    # Verification
    with open(output_path, "rb") as f:
        loaded = pickle.load(f)
    print(f"\n  Verification:")
    print(f"    mu shape:        {loaded['mu'].shape}")
    print(f"    sigma_inv shape: {loaded['sigma_inv'].shape}")
    print(f"    priors:          {loaded['priors']}")

    return lda_params


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export LDA parameters for online Adaptive LDA"
    )
    parser.add_argument(
        "--subjects", "-s",
        type=int, nargs="+",
        default=list(range(1, 16)),
        help="Subject IDs to use (default: 1-15)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path (default: resources/models/lda_params.pkl)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Config path (default: resources/configs/bci_config.yaml)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Regularization parameter (default: 1e-6)"
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=0.5,
        help="Epoch start time in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=3.5,
        help="Epoch end time in seconds (default: 3.5)"
    )

    args = parser.parse_args()

    export_lda_params(
        subjects=args.subjects,
        output_path=args.output,
        config_path=args.config,
        eps=args.eps,
        tmin=args.tmin,
        tmax=args.tmax
    )


if __name__ == "__main__":
    main()

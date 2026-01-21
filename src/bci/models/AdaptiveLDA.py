"""
Adaptive LDA Classifier for Motor Imagery BCI.

Based on: Wu et al. (2024) "Adaptive LDA Classifier Enhances Real-Time Control
of an EEG Brain-Computer Interface for Decoding Imagined Syllables"
Brain Sciences, 14(3), 196.

This model implements an adaptive Linear Discriminant Analysis classifier that:
- Extracts PSD features from EEG signals (8-30 Hz for MI, 2 Hz steps)
- Uses LDA for classification
- Adapts online by updating class means and covariance matrix
- Handles non-stationary EEG signals during real-time BCI control

Markers used:
- 0: unknown
- 1: rest
- 2: left_hand
- 3: right_hand

Parameters are read from bci_config.yaml:
- fs: sampling frequency
- frequencies: [low, high] for feature extraction
"""

import pickle
import numpy as np

from .AdaptiveLDA_modules.feature_extraction import extract_psd_features
from .AdaptiveLDA_modules.lda_core import LDACore
from .AdaptiveLDA_modules.adaptive_update import (
    update_mean,
    update_covariance_inv_woodbury
)


class AdaptiveLDA:
    """
    Adaptive LDA classifier with online parameter updates.

    This classifier adapts to non-stationary EEG signals by continuously
    updating its parameters (class means and covariance) based on new data.

    Parameters:
    -----------
    uc_mu : float
        Update coefficient for class means (default: 0.4 * 2^-6 ≈ 0.00625)
        Lower values = slower adaptation
    uc_sigma : float
        Update coefficient for covariance (default: 0.4 * 2^-3 = 0.05)
        Lower values = slower adaptation
    sfreq : float
        Sampling frequency in Hz (from config.fs)
    freq_range : tuple
        (min_freq, max_freq) in Hz (from config.frequencies)
    freq_step : int
        Frequency step for PSD extraction in Hz
    """

    def __init__(self, uc_mu=0.4 * 2**-6, uc_sigma=0.4 * 2**-3,
                 sfreq=None, freq_range=None, freq_step=2):
        """
        Initialize the Adaptive LDA classifier.

        Parameters from config should be passed via model_args dict:
        - sfreq: from config.fs
        - freq_range: from config.frequencies (tuple of [low, high])
        """
        # Update coefficients (optimized from paper)
        self.uc_mu = uc_mu
        self.uc_sigma = uc_sigma

        # Feature extraction parameters
        # If not provided, use defaults from bci_config.yaml
        self.sfreq = sfreq if sfreq is not None else 160.0
        self.freq_range = freq_range if freq_range is not None else (8.0, 30.0)
        self.freq_step = freq_step

        # LDA classifier core
        self.lda = LDACore()

        # For compatibility with existing codebase
        self.classes = None
        self.n_features = None

    def fit(self, signals, y):
        """
        Train the classifier on EEG signals.

        Parameters:
        -----------
        signals : np.ndarray
            Shape (n_trials, n_channels, n_samples)
            Raw EEG signals (already filtered)
        y : np.ndarray
            Shape (n_trials,)
            Class labels (1: rest, 2: left_hand, 3: right_hand)
        """
        # Extract PSD features
        X = extract_psd_features(
            signals,
            sfreq=self.sfreq,
            freq_range=self.freq_range,
            freq_step=self.freq_step
        )

        # Train LDA classifier
        self.lda.fit(X, y)

        # Store metadata
        self.classes = self.lda.classes
        self.n_features = X.shape[1]

        return self

    def predict(self, signals):
        """
        Predict class labels for EEG signals.

        Parameters:
        -----------
        signals : np.ndarray
            Shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
            Raw EEG signals

        Returns:
        --------
        predictions : np.ndarray
            Predicted class labels
        """
        # Extract features
        X = extract_psd_features(
            signals,
            sfreq=self.sfreq,
            freq_range=self.freq_range,
            freq_step=self.freq_step
        )

        # Predict
        return self.lda.predict(X)

    def predict_proba(self, signals):
        """
        Predict class probabilities for EEG signals.

        Parameters:
        -----------
        signals : np.ndarray
            Shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
            Raw EEG signals

        Returns:
        --------
        probabilities : np.ndarray
            Shape (n_trials, n_classes) or (1, n_classes)
            Class probabilities for each trial
        """
        # Extract features
        X = extract_psd_features(
            signals,
            sfreq=self.sfreq,
            freq_range=self.freq_range,
            freq_step=self.freq_step
        )

        # Predict probabilities
        return self.lda.predict_proba(X)

    def update(self, signal, y_true):
        """
        Update classifier parameters with new trial (online adaptation).

        This is called during online control to adapt the classifier to
        non-stationary EEG signals and user learning.

        Parameters:
        -----------
        signal : np.ndarray
            Shape (n_channels, n_samples)
            Single trial of EEG data
        y_true : int
            True class label for this trial
        """
        # Extract features from new trial
        x_new = extract_psd_features(
            signal,
            sfreq=self.sfreq,
            freq_range=self.freq_range,
            freq_step=self.freq_step
        )

        # Save old mean for covariance update
        mu_old = self.lda.mu[y_true].copy()

        # Update class mean (Equation 4)
        self.lda.mu[y_true] = update_mean(
            mu_old, x_new, self.uc_mu
        )

        # Update inverse covariance using Woodbury identity (Equations 5-11)
        self.lda.sigma_inv = update_covariance_inv_woodbury(
            self.lda.sigma_inv,
            x_new,
            mu_old,
            self.uc_sigma,
            len(self.classes)
        )

        # Update LDA parameters (w and b)
        self.lda._update_parameters()

    def save(self, filepath):
        """
        Save the classifier to disk.

        Parameters:
        -----------
        filepath : str
            Path where to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """
        Load a classifier from disk.

        Parameters:
        -----------
        filepath : str
            Path to saved model

        Returns:
        --------
        model : AdaptiveLDA
            Loaded classifier
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

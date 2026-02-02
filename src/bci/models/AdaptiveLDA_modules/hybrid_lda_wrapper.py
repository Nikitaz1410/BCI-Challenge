"""
Wrapper for HybridLDA with internal feature extraction.

This wrapper allows HybridLDA to work like AllRounderBCIModel:
- Accepts raw signals (N, C, T) instead of pre-extracted features
- Supports multiple feature extractors (CSP, Welch bandpower, log-bandpower, etc.)
- Same API as baseline models for easy comparison
"""

import numpy as np
from typing import Union, Sequence, Dict, Any, Optional
from .hybrid_lda import HybridLDA
from bci.Models.Baseline import (
    CSPFeatureExtractor,
    WelchBandPowerExtractor,
    ConcatFeatureUnion,
    _build_feature_extractor,
    _normalize_feature_list
)
from .feature_extraction import extract_log_bandpower_features


class LogBandPowerExtractor:
    """Log-bandpower extractor for HybridLDA (mu and beta bands)."""
    
    def __init__(self, sfreq: float, mu_band=(8, 12), beta_band=(13, 30)):
        self.sfreq = sfreq
        self.mu_band = mu_band
        self.beta_band = beta_band
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Stateless extractor - no fitting needed."""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract log-bandpower features from signals."""
        return extract_log_bandpower_features(
            X, sfreq=self.sfreq,
            mu_band=self.mu_band,
            beta_band=self.beta_band
        )


class HybridLDAWrapper:
    """
    Wrapper for HybridLDA with internal feature extraction.
    
    This allows HybridLDA to work like AllRounderBCIModel:
    - Accepts raw signals (N, C, T) instead of pre-extracted features
    - Supports multiple feature extractors (CSP, Welch bandpower, log-bandpower, etc.)
    - Same API as baseline models for easy comparison
    
    Parameters:
    -----------
    features : str, list, or dict
        Feature extractor(s) to use. Same format as AllRounderBCIModel:
        - "csp", "welch_bandpower", "log_bandpower"
        - ["csp", "welch_bandpower"]
        - [{"name": "csp", "n_components": 8}, ...]
    
    move_threshold : float
        Confidence threshold for HybridLDA Stage B (default: 0.5)
    
    reg : float
        Regularization for LDA (default: 1e-2)
    
    shrinkage_alpha : float or None
        Shrinkage parameter for LDA
    
    uc_mu : float
        Update coefficient for online adaptation (default: 0.4 * 2**-6)
    
    sfreq : float
        Sampling frequency (required for bandpower features)
    
    use_improved_composition : bool
        If True, uses confidence-weighted Stage B probabilities instead of simple multiplication.
        When Stage B is uncertain, blends toward uniform (0.5/0.5) distribution.
        This produces better calibrated probabilities and reduces overconfidence (default: True).
    """
    
    def __init__(
        self,
        *,
        features: Union[str, Sequence[Union[str, Dict[str, Any]]], None] = None,
        move_threshold: float = 0.5,
        reg: float = 1e-2,
        shrinkage_alpha: Optional[float] = 0.1,
        uc_mu: float = 0.4 * 2**-6,
        sfreq: Optional[float] = None,
        use_improved_composition: bool = True,
    ):
        # Default to log-bandpower if no features specified
        if features is None:
            features = ["log_bandpower"]
        
        self.feature_specs = _normalize_feature_list(features)
        self.move_threshold = move_threshold
        self.reg = reg
        self.shrinkage_alpha = shrinkage_alpha
        self.uc_mu = uc_mu
        self.sfreq = sfreq
        self.use_improved_composition = use_improved_composition
        
        self._feature_extractor: Optional[ConcatFeatureUnion] = None
        self._hybrid_lda: Optional[HybridLDA] = None
        self._fitted: bool = False
        self._meta: Dict[str, Any] = {}
    
    def _build_custom_extractors(self, specs):
        """Build extractors, handling custom 'log_bandpower' type."""
        extractors = []
        for spec in specs:
            if isinstance(spec, str):
                name = spec.lower().strip()
                params = {}
            else:
                name = str(spec.get("name", "")).lower().strip()
                params = dict(spec)
                params.pop("name", None)
            
            # Handle log_bandpower specially
            if name == "log_bandpower":
                sfreq = params.get("sfreq", self.sfreq)
                mu_band = params.get("mu_band", (8, 12))
                beta_band = params.get("beta_band", (13, 30))
                
                if sfreq is None:
                    raise ValueError("sfreq must be provided for log_bandpower features")
                
                extractors.append(LogBandPowerExtractor(sfreq, mu_band, beta_band))
            else:
                # Use baseline extractors (CSP, Welch bandpower, etc.)
                # Pass sfreq to extractors that need it
                if self.sfreq is not None:
                    if isinstance(spec, dict):
                        spec = dict(spec)  # Make a copy
                        if "sfreq" not in spec:
                            spec["sfreq"] = self.sfreq
                    else:
                        # Convert string to dict to add sfreq
                        spec = {"name": spec, "sfreq": self.sfreq}
                extractors.append(_build_feature_extractor(spec))
        
        return extractors
    
    def fit(self, signals: np.ndarray, labels: np.ndarray) -> "HybridLDAWrapper":
        """
        Fit the model on raw signals.
        
        Parameters:
        -----------
        signals : np.ndarray
            Shape (N, C, T) - raw EEG signals
        labels : np.ndarray
            Shape (N,) - class labels (0=rest, 1=left, 2=right)
        """
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError(f"Signals must have shape [N, C, T], got shape {signals.shape}")
        
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError(f"Labels must be 1D, got shape {labels.shape}")
        
        # Build feature extractors
        extractors = self._build_custom_extractors(self.feature_specs)
        self._feature_extractor = ConcatFeatureUnion(extractors)
        
        # Extract features
        X_features = self._feature_extractor.fit_transform(signals, labels)
        
        # Train HybridLDA on extracted features
        self._hybrid_lda = HybridLDA(
            move_threshold=self.move_threshold,
            reg=self.reg,
            shrinkage_alpha=self.shrinkage_alpha,
            uc_mu=self.uc_mu,
            use_improved_composition=self.use_improved_composition
        )
        self._hybrid_lda.fit(X_features, labels)
        
        self._meta = {
            "n_samples": int(signals.shape[0]),
            "n_channels": int(signals.shape[1]),
            "n_timepoints": int(signals.shape[2]),
            "n_features": int(X_features.shape[1]),
            "feature_specs": self.feature_specs,
        }
        
        self._fitted = True
        return self
    
    def _ensure_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model is not trained. Call `fit` first.")
    
    def predict(self, signals: np.ndarray) -> np.ndarray:
        """Predict class labels from raw signals."""
        self._ensure_fitted()
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError(f"Signals must have shape [N, C, T], got shape {signals.shape}")
        
        # Extract features
        X_features = self._feature_extractor.transform(signals)
        
        # Predict with HybridLDA
        return self._hybrid_lda.predict(X_features)
    
    def predict_proba(self, signals: np.ndarray) -> np.ndarray:
        """Predict class probabilities from raw signals."""
        self._ensure_fitted()
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError(f"Signals must have shape [N, C, T], got shape {signals.shape}")
        
        # Extract features
        X_features = self._feature_extractor.transform(signals)
        
        # Predict probabilities with HybridLDA
        return self._hybrid_lda.predict_proba(X_features)
    
    def update(self, label: int, signal: np.ndarray):
        """
        Online adaptation: update model with new labeled sample.
        
        Parameters:
        -----------
        label : int
            True label (0=rest, 1=left, 2=right)
        signal : np.ndarray
            Shape (C, T) or (1, C, T) - single signal sample
        """
        self._ensure_fitted()
        
        # Handle single sample
        if signal.ndim == 2:
            signal = signal[np.newaxis, :, :]
        
        # Extract features
        features = self._feature_extractor.transform(signal)
        
        # Update HybridLDA
        self._hybrid_lda.update(label, features[0])
    
    def save(self, path: str) -> str:
        """Save the model."""
        import pickle
        import os
        self._ensure_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        checkpoint = {
            "feature_specs": self.feature_specs,
            "feature_extractor": self._feature_extractor,
            "hybrid_lda": self._hybrid_lda,
            "meta": self._meta,
            "move_threshold": self.move_threshold,
            "reg": self.reg,
            "shrinkage_alpha": self.shrinkage_alpha,
            "uc_mu": self.uc_mu,
            "sfreq": self.sfreq,
            "use_improved_composition": self.use_improved_composition,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        return path
    
    @classmethod
    def load(cls, path: str) -> "HybridLDAWrapper":
        """Load a saved model."""
        import pickle
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        
        instance = cls(
            features=checkpoint.get("feature_specs", None),
            move_threshold=checkpoint.get("move_threshold", 0.5),
            reg=checkpoint.get("reg", 1e-2),
            shrinkage_alpha=checkpoint.get("shrinkage_alpha", 0.1),
            uc_mu=checkpoint.get("uc_mu", 0.4 * 2**-6),
            sfreq=checkpoint.get("sfreq", None),
            use_improved_composition=checkpoint.get("use_improved_composition", True),  # Default to True for backward compatibility
        )
        instance._feature_extractor = checkpoint["feature_extractor"]
        instance._hybrid_lda = checkpoint["hybrid_lda"]
        instance._meta = checkpoint.get("meta", {})
        instance._fitted = True
        return instance

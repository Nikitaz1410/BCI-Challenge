"""
Wrapper for CombinedAdaptiveLDA with internal feature extraction.

This wrapper allows CombinedAdaptiveLDA to work like AllRounderBCIModel:
- Accepts raw signals (N, C, T) instead of pre-extracted features
- Supports multiple feature extractors (CSP, Welch bandpower, log-bandpower, etc.)
- Same API as baseline models for easy comparison
"""

import numpy as np
from typing import Union, Sequence, Dict, Any, Optional
from .combined_adaptive_lda import CombinedAdaptiveLDA
from bci.models.Baseline import (
    CSPFeatureExtractor,
    WelchBandPowerExtractor,
    ConcatFeatureUnion,
    _build_feature_extractor,
    _normalize_feature_list
)
from .feature_extraction import extract_log_bandpower_features
from .hybrid_lda_wrapper import LogBandPowerExtractor


class CombinedAdaptiveLDAWrapper:
    """
    Wrapper for CombinedAdaptiveLDA with internal feature extraction.
    
    Parameters:
    -----------
    features : str, list, or dict
        Feature extractor(s) to use. Same format as AllRounderBCIModel.
    
    confidence_threshold : float
        Threshold for model selection (default: 0.7)
    
    ensemble_weight : float
        Weight for ensemble (default: 0.5)
    
    move_threshold : float
        Threshold for HybridLDA Stage B (default: 0.6)
    
    reg : float
        Regularization for both models (default: 1e-3)
    
    shrinkage_alpha : float or None
        Shrinkage parameter for both models
    
    uc_mu : float
        Update coefficient for online adaptation (default: 0.4 * 2**-7)
    
    use_adaptive_lr : bool
        Use adaptive learning rate (default: True)
    
    sfreq : float
        Sampling frequency (required for bandpower features)
    
    use_improved_composition : bool
        Use improved probability composition in HybridLDA (default: True)
    """
    
    def __init__(
        self,
        *,
        features: Union[str, Sequence[Union[str, Dict[str, Any]]], None] = None,
        confidence_threshold: float = 0.7,
        ensemble_weight: float = 0.5,
        move_threshold: float = 0.6,
        reg: float = 1e-3,
        shrinkage_alpha: Optional[float] = 0.05,
        uc_mu: float = 0.4 * 2**-7,
        use_adaptive_lr: bool = True,
        sfreq: Optional[float] = None,
        use_improved_composition: bool = True,
    ):
        # Default to log-bandpower if no features specified
        if features is None:
            features = ["log_bandpower"]
        
        self.feature_specs = _normalize_feature_list(features)
        self.confidence_threshold = confidence_threshold
        self.ensemble_weight = ensemble_weight
        self.move_threshold = move_threshold
        self.reg = reg
        self.shrinkage_alpha = shrinkage_alpha
        self.uc_mu = uc_mu
        self.use_adaptive_lr = use_adaptive_lr
        self.sfreq = sfreq
        self.use_improved_composition = use_improved_composition
        
        self._feature_extractor: Optional[ConcatFeatureUnion] = None
        self._combined_lda: Optional[CombinedAdaptiveLDA] = None
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
                if self.sfreq is not None:
                    if isinstance(spec, dict):
                        spec = dict(spec)
                        if "sfreq" not in spec:
                            spec["sfreq"] = self.sfreq
                    else:
                        spec = {"name": spec, "sfreq": self.sfreq}
                extractors.append(_build_feature_extractor(spec))
        
        return extractors
    
    def fit(self, signals: np.ndarray, labels: np.ndarray) -> "CombinedAdaptiveLDAWrapper":
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
        
        # Train CombinedAdaptiveLDA on extracted features
        self._combined_lda = CombinedAdaptiveLDA(
            confidence_threshold=self.confidence_threshold,
            ensemble_weight=self.ensemble_weight,
            move_threshold=self.move_threshold,
            reg=self.reg,
            shrinkage_alpha=self.shrinkage_alpha,
            uc_mu=self.uc_mu,
            use_adaptive_lr=self.use_adaptive_lr,
            use_improved_composition=self.use_improved_composition
        )
        self._combined_lda.fit(X_features, labels)
        
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
        
        # Predict with CombinedAdaptiveLDA
        return self._combined_lda.predict(X_features)
    
    def predict_proba(self, signals: np.ndarray) -> np.ndarray:
        """Predict class probabilities from raw signals."""
        self._ensure_fitted()
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError(f"Signals must have shape [N, C, T], got shape {signals.shape}")
        
        # Extract features
        X_features = self._feature_extractor.transform(signals)
        
        # Predict probabilities with CombinedAdaptiveLDA
        return self._combined_lda.predict_proba(X_features)
    
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
        
        # Update CombinedAdaptiveLDA
        self._combined_lda.update(label, features[0])
    
    def save(self, path: str) -> str:
        """Save the model."""
        import pickle
        import os
        self._ensure_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        checkpoint = {
            "feature_specs": self.feature_specs,
            "feature_extractor": self._feature_extractor,
            "combined_lda": self._combined_lda,
            "meta": self._meta,
            "confidence_threshold": self.confidence_threshold,
            "ensemble_weight": self.ensemble_weight,
            "move_threshold": self.move_threshold,
            "reg": self.reg,
            "shrinkage_alpha": self.shrinkage_alpha,
            "uc_mu": self.uc_mu,
            "use_adaptive_lr": self.use_adaptive_lr,
            "sfreq": self.sfreq,
            "use_improved_composition": self.use_improved_composition,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        return path
    
    @classmethod
    def load(cls, path: str) -> "CombinedAdaptiveLDAWrapper":
        """Load a saved model."""
        import pickle
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        
        instance = cls(
            features=checkpoint.get("feature_specs", None),
            confidence_threshold=checkpoint.get("confidence_threshold", 0.7),
            ensemble_weight=checkpoint.get("ensemble_weight", 0.5),
            move_threshold=checkpoint.get("move_threshold", 0.6),
            reg=checkpoint.get("reg", 1e-3),
            shrinkage_alpha=checkpoint.get("shrinkage_alpha", 0.05),
            uc_mu=checkpoint.get("uc_mu", 0.4 * 2**-7),
            use_adaptive_lr=checkpoint.get("use_adaptive_lr", True),
            sfreq=checkpoint.get("sfreq", None),
            use_improved_composition=checkpoint.get("use_improved_composition", True),
        )
        instance._feature_extractor = checkpoint["feature_extractor"]
        instance._combined_lda = checkpoint["combined_lda"]
        instance._meta = checkpoint.get("meta", {})
        instance._fitted = True
        return instance

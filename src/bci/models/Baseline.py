"""
Baseline / classic ML model wrappers.

This file provides a modular wrapper that can combine multiple feature extraction
techniques and multiple classic classifiers behind a single API:

- fit()
- predict()
- predict_proba()
- save()/load()

Supported feature extractors (configurable and composable):
- CSP (binary or multi-class via one-vs-rest)
- Welch band-power (log band power per channel/band)
- Lateralization features (hemispheric differences/ratios, configurable)

Supported classifiers:
- 'svm', 'lda', 'logreg'/'lr', 'rf'
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from mne.decoding import CSP
from scipy import signal as scipy_signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from bci.utils.bci_config import load_config


def _load_config_values():
    """Load default values from the config file."""
    try:
        # Try to find config relative to common project structure
        current_file = Path(__file__).resolve()
        # Navigate up to project root (src/bci/models -> project root)
        project_root = current_file.parent.parent.parent.parent
        config_path = project_root / "resources" / "configs" / "bci_config.yaml"
        
        if config_path.exists():
            config = load_config(config_path)
            return config.fs, config.window_size, config.step_size
    except Exception:
        pass
    
    # Fallback defaults
    return 250.0, 250, 32


def _get_default_sfreq() -> float:
    """Load the default sampling frequency from the config file."""
    return _load_config_values()[0]


def _get_default_window_size() -> int:
    """Load the default window size from the config file."""
    return _load_config_values()[1]


def _get_default_step_size() -> int:
    """Load the default step size from the config file."""
    return _load_config_values()[2]


class MultiClassCSP:
    """
    Multi-class CSP using one-vs-rest strategy.
    Creates CSP filters for each class vs all others using MNE's CSP.
    """
    
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.csp_list_ = []
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiClassCSP':
        """
        Fit CSP filters for each class (one-vs-rest).
        """
        self.classes_ = np.unique(y)
        self.csp_list_ = []
        
        for cls in self.classes_:
            # Binary labels: current class vs rest
            y_binary = (y == cls).astype(int)
            
            csp = CSP(n_components=self.n_components, log=True, norm_trace=True)
            csp.fit(X, y_binary)
            self.csp_list_.append(csp)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply all CSP filters and concatenate features.
        """
        features_list = []
        for csp in self.csp_list_:
            features_list.append(csp.transform(X))
        
        return np.concatenate(features_list, axis=1)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class CSPFeatureExtractor:
    """CSP feature extractor (optionally multi-class via one-vs-rest)."""
    
    def __init__(
        self,
        n_csp_components: int = 6,
        use_multiclass_csp: bool = True,
    ):
        self.n_csp_components = n_csp_components
        self.use_multiclass_csp = use_multiclass_csp

        self.csp_ = None
        self.fitted_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPFeatureExtractor":
        """Fit CSP filters."""
        if self.use_multiclass_csp:
            self.csp_ = MultiClassCSP(n_components=self.n_csp_components)
        else:
            self.csp_ = CSP(n_components=self.n_csp_components, log=True, norm_trace=True)
        
        self.csp_.fit(X, y)
        self.fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return CSP features shaped (n_epochs, n_features)."""
        if not self.fitted_:
            raise RuntimeError("CSPFeatureExtractor must be fitted before transform")
        
        return self.csp_.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)


class WelchBandPowerExtractor:
    """Welch log band-power per channel and band."""

    def __init__(
        self,
        sfreq: Optional[float] = None,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        eps: float = 1e-10,
    ) -> None:
        self.sfreq = sfreq if sfreq is not None else _get_default_sfreq()
        self.freq_bands = freq_bands or {"mu": (8, 12), "beta": (13, 30)}
        self.window_size = window_size if window_size is not None else _get_default_window_size()
        self.step_size = step_size if step_size is not None else _get_default_step_size()
        self.eps = float(eps)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "WelchBandPowerExtractor":
        # Stateless
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Signals must have shape [N, C, T], got shape {X.shape}")

        n_epochs, n_channels, n_samples = X.shape
        bands: List[Tuple[str, Tuple[float, float]]] = list(self.freq_bands.items())
        n_bands = len(bands)
        features = np.zeros((n_epochs, n_channels * n_bands), dtype=np.float64)

        for epoch_idx in range(n_epochs):
            feat_idx = 0
            for ch_idx in range(n_channels):
                nperseg = min(self.window_size, n_samples)
                noverlap = min(self.window_size - self.step_size, n_samples - self.step_size)
                noverlap = max(0, noverlap)

                freqs, psd = scipy_signal.welch(
                    X[epoch_idx, ch_idx],
                    fs=self.sfreq,
                    nperseg=nperseg,
                    noverlap=noverlap,
                )

                for _, (low, high) in bands:
                    band_mask = (freqs >= low) & (freqs <= high)
                    band_power = float(np.mean(psd[band_mask])) if band_mask.any() else self.eps
                    features[epoch_idx, feat_idx] = np.log(band_power + self.eps)
                    feat_idx += 1

        return features


class LateralizationExtractor:
    """
    Lateralization features from left/right channel groups.

    This is intentionally channel-index based (not montage-name based) to keep it
    dataset-agnostic. You provide indices for left and right hemispheres.

    Features produced per epoch:
    - mean(left) and mean(right) for each base feature group (optional)
    - difference: left - right (default)
    - ratio: (left + eps) / (right + eps) (optional)

    The "base features" are computed as Welch band-power by default.
    """

    def __init__(
        self,
        left_indices: Sequence[int],
        right_indices: Sequence[int],
        *,
        sfreq: Optional[float] = None,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        include_means: bool = False,
        include_ratio: bool = False,
        eps: float = 1e-10,
    ) -> None:
        self.left_indices = tuple(int(i) for i in left_indices)
        self.right_indices = tuple(int(i) for i in right_indices)
        if len(self.left_indices) == 0 or len(self.right_indices) == 0:
            raise ValueError("left_indices and right_indices must be non-empty")

        self.include_means = bool(include_means)
        self.include_ratio = bool(include_ratio)
        self.eps = float(eps)

        self._bp = WelchBandPowerExtractor(
            sfreq=sfreq,
            freq_bands=freq_bands,
            window_size=window_size,
            step_size=step_size,
            eps=eps,
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LateralizationExtractor":
        self._bp.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Signals must have shape [N, C, T], got shape {X.shape}")

        n_epochs, n_channels, _ = X.shape
        if max(self.left_indices + self.right_indices) >= n_channels:
            raise ValueError(
                f"Channel index out of range for n_channels={n_channels}. "
                f"left_indices={self.left_indices}, right_indices={self.right_indices}"
            )

        # Compute log band power per channel/band, then reduce over hemispheres.
        # Shape: (N, C * B) -> reshape to (N, C, B)
        bp = self._bp.transform(X)
        n_bands = len(self._bp.freq_bands)
        bp = bp.reshape(n_epochs, n_channels, n_bands)

        left = bp[:, self.left_indices, :].mean(axis=1)   # (N, B)
        right = bp[:, self.right_indices, :].mean(axis=1)  # (N, B)
        diff = left - right

        feats = [diff]
        if self.include_means:
            feats = [left, right] + feats
        if self.include_ratio:
            ratio = (left + self.eps) / (right + self.eps)
            feats.append(ratio)

        return np.concatenate(feats, axis=1)


class ConcatFeatureUnion:
    """Fit/transform a list of extractors and concatenate their outputs."""

    def __init__(self, extractors: Sequence[Any]) -> None:
        if not extractors:
            raise ValueError("At least one feature extractor must be provided")
        self.extractors = list(extractors)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ConcatFeatureUnion":
        for ex in self.extractors:
            # Support fit(X, y) and fit(X) styles
            if hasattr(ex, "fit"):
                try:
                    ex.fit(X, y)
                except TypeError:
                    ex.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats: List[np.ndarray] = []
        for ex in self.extractors:
            if not hasattr(ex, "transform"):
                raise TypeError(f"Extractor {type(ex).__name__} has no transform()")
            f = ex.transform(X)
            f = np.asarray(f)
            if f.ndim != 2:
                raise ValueError(
                    f"Extractor {type(ex).__name__} must return 2D features (N, F), got {f.shape}"
                )
            feats.append(f)
        return np.concatenate(feats, axis=1) if len(feats) > 1 else feats[0]

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


def _normalize_feature_list(
    features: Union[str, Sequence[Union[str, Dict[str, Any]]], None],
) -> List[Union[str, Dict[str, Any]]]:
    """
    Normalize user-provided feature config into a list.

    Accepted forms:
    - "csp"
    - ["csp", "welch_bandpower"]
    - [{"name": "csp", ...}, {"name": "welch_bandpower", ...}]
    """
    if features is None:
        return ["csp"]
    if isinstance(features, (str, bytes)):
        return [str(features)]
    # Allow a single dict spec (common ergonomic usage)
    if isinstance(features, Mapping):
        return [dict(features)]
    return list(features)


def _build_feature_extractor(spec: Union[str, Dict[str, Any]]) -> Any:
    """
    Build a feature extractor from a name or dict spec.

    Names:
    - 'csp', 'multiclass_csp'
    - 'welch_bandpower', 'bandpower'
    - 'lateralization'
    """
    if isinstance(spec, str):
        name = spec
        params: Dict[str, Any] = {}
    else:
        name = str(spec.get("name", ""))
        params = dict(spec)
        params.pop("name", None)

    name = name.lower().strip()

    if name in ("csp",):
        return CSPFeatureExtractor(
            n_csp_components=int(params.get("n_components", params.get("n_csp_components", 6))),
            use_multiclass_csp=bool(params.get("use_multiclass_csp", True)),
        )
    if name in ("multiclass_csp", "mcsp", "multi_csp"):
        return CSPFeatureExtractor(
            n_csp_components=int(params.get("n_components", params.get("n_csp_components", 6))),
            use_multiclass_csp=True,
        )
    if name in ("welch_bandpower", "bandpower", "welch_bp"):
        return WelchBandPowerExtractor(
            sfreq=params.get("sfreq"),
            freq_bands=params.get("freq_bands"),
            window_size=params.get("window_size"),
            step_size=params.get("step_size"),
        )
    if name in ("lateralization", "lateralisation", "lat"):
        return LateralizationExtractor(
            left_indices=params["left_indices"],
            right_indices=params["right_indices"],
            sfreq=params.get("sfreq"),
            freq_bands=params.get("freq_bands"),
            window_size=params.get("window_size"),
            step_size=params.get("step_size"),
            include_means=bool(params.get("include_means", False)),
            include_ratio=bool(params.get("include_ratio", False)),
        )

    raise ValueError(
        f"Unsupported feature extractor '{name}'. "
        "Expected one of ['csp', 'multiclass_csp', 'welch_bandpower', 'lateralization']."
    )


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _as_int_labels(labels: Optional[np.ndarray], fallback_size: int) -> np.ndarray:
    """
    Convert labels to a 1D integer array.

    If labels is None, return zeros of length `fallback_size`.
    Accepts either class indices (shape [N]) or one-hot arrays (shape [N, C]).
    """
    if labels is None:
        return np.zeros(fallback_size, dtype=np.int64)

    labels = np.asarray(labels)
    if labels.ndim == 1:
        return labels.astype(np.int64)
    if labels.ndim == 2:
        return labels.argmax(axis=1).astype(np.int64)

    raise ValueError("Labels must be 1D class indices or 2D one-hot arrays.")


def _create_classifier(classifier_type: str) -> Any:
    """
    Create a classifier instance based on the classifier type string.

    Args:
        classifier_type: One of 'svm', 'lda', 'logreg', 'lr', 'rf'

    Returns:
        A scikit-learn classifier instance.
    """
    classifier_type = str(classifier_type or "lda").lower()

    if classifier_type in ("lda", "csp-lda", "linear_discriminant"):
        # Shrinkage LDA with auto-regularization
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    elif classifier_type in ("svm", "svc", "csp-svm"):
        return SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced'
        )
    elif classifier_type in ("logreg", "lr", "logistic", "logistic_regression", "csp-logreg", "csp-lr"):
        return LogisticRegression(max_iter=2000, class_weight=None)
    elif classifier_type in ("rf", "random_forest", "csp-rf"):
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
    else:
        raise ValueError(
            f"Unsupported classifier_type '{classifier_type}'. "
            "Expected one of ['lda', 'svm', 'logreg', 'rf']."
        )


class AllRounderBCIModel:
    """
    Modular allrounder model wrapper.

    Parameters
    ----------
    features:
        Feature extractor(s) to use. Can be a string, list of strings,
        or list of dict specs.

        Examples:
            features="csp"
            features=["csp", "welch_bandpower"]
            features=[
                {"name": "csp", "n_components": 8, "use_multiclass_csp": True},
                {"name": "welch_bandpower", "freq_bands": {"mu": (8, 12), "beta": (13, 30)}},
                {"name": "lateralization", "left_indices": [0,1], "right_indices":[2,3]},
            ]

    classifier:
        One of 'svm', 'lda', 'logreg'/'lr', 'rf'.

    scale:
        Whether to standardize concatenated features before classification.
    """

    def __init__(
        self,
        *,
        features: Union[str, Sequence[Union[str, Dict[str, Any]]], None] = None,
        classifier: str = "lda",
        scale: bool = True,
    ) -> None:
        self.feature_specs = _normalize_feature_list(features)
        self.classifier_type = str(classifier or "lda")
        self.scale = bool(scale)

        self._feature_extractor: Optional[ConcatFeatureUnion] = None
        self._scaler: Optional[StandardScaler] = None
        self._classifier: Optional[Any] = None
        self._classes: Optional[np.ndarray] = None
        self._meta: Dict[str, Any] = {}
        self._fitted: bool = False

    def fit(self, signals: np.ndarray, labels: np.ndarray) -> "AllRounderBCIModel":
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError(f"Signals must have shape [N, C, T], got shape {signals.shape}")

        labels = _as_int_labels(labels, fallback_size=signals.shape[0])
        self._classes = np.unique(labels)

        extractors = [_build_feature_extractor(s) for s in self.feature_specs]
        self._feature_extractor = ConcatFeatureUnion(extractors)

        X = self._feature_extractor.fit_transform(signals, labels)

        if self.scale:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        else:
            self._scaler = None

        self._classifier = _create_classifier(self.classifier_type)
        self._classifier.fit(X, labels)

        self._meta = {
            "n_samples": int(signals.shape[0]),
            "n_channels": int(signals.shape[1]),
            "n_timepoints": int(signals.shape[2]),
            "n_classes": int(len(self._classes)),
            "classifier_type": self.classifier_type,
            "feature_specs": self.feature_specs,
            "scale": self.scale,
        }

        self._fitted = True
        return self

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not trained. Call `fit` or `load` first.")

    def predict(self, signals: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError(f"Signals must have shape [N, C, T], got shape {signals.shape}")

        X = self._feature_extractor.transform(signals)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._classifier.predict(X)

    def predict_proba(self, signals: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError(f"Signals must have shape [N, C, T], got shape {signals.shape}")

        X = self._feature_extractor.transform(signals)
        if self._scaler is not None:
            X = self._scaler.transform(X)

        if hasattr(self._classifier, "predict_proba"):
            return self._classifier.predict_proba(X)

        # Fallback: convert decision function to probabilities.
        decision = self._classifier.decision_function(X)
        if decision.ndim == 1:
            decision = decision.reshape(-1, 1)
        exp_dec = np.exp(decision - decision.max(axis=1, keepdims=True))
        return exp_dec / exp_dec.sum(axis=1, keepdims=True)

    def save(self, path: str) -> str:
        self._ensure_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        checkpoint = {
            "feature_specs": self.feature_specs,
            "feature_extractor": self._feature_extractor,
            "scaler": self._scaler,
            "classifier": self._classifier,
            "classes": self._classes,
            "meta": self._meta,
            "classifier_type": self.classifier_type,
            "scale": self.scale,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        return path

    @classmethod
    def load(cls, path: str) -> "AllRounderBCIModel":
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        instance = cls(
            features=checkpoint.get("feature_specs", None),
            classifier=checkpoint.get("classifier_type", "lda"),
            scale=checkpoint.get("scale", True),
        )
        instance._feature_extractor = checkpoint["feature_extractor"]
        instance._scaler = checkpoint.get("scaler", None)
        instance._classifier = checkpoint["classifier"]
        instance._classes = checkpoint.get("classes", None)
        instance._meta = checkpoint.get("meta", {})
        instance._fitted = True
        return instance

"""
Combined Adaptive LDA: Hybrid LDA + Core LDA with Adaptive Selection

This model combines the strengths of both HybridLDA (2-stage hierarchical) and 
Core LDA (standard 3-class) with adaptive selection based on confidence.

Inspired by: "Adaptive LDA Classifier Enhances Real-Time Control of an EEG
Brain–Computer Interface for Imagined-Syllables Decoding" (Wu et al.)

Key Features:
1. Confidence-based model selection: Use Core LDA when confident, HybridLDA when uncertain
2. Adaptive mean updates: Both models adapt online using EMA
3. Ensemble predictions: Weighted combination of both models
4. Adaptive learning rates: Adjust update rates based on prediction confidence
"""

import numpy as np
from typing import Dict, Any, Optional
from .hybrid_lda import HybridLDA
from .lda_core import LDACore


class CombinedAdaptiveLDA:
    """
    Combined Adaptive LDA that intelligently switches between HybridLDA and Core LDA.
    
    Strategy:
    - When predictions are confident (high max probability), use Core LDA (simpler, faster)
    - When predictions are uncertain (low max probability), use HybridLDA (more robust)
    - Ensemble both predictions with confidence-based weighting
    
    Parameters:
    -----------
    confidence_threshold : float
        Threshold for model selection (default: 0.7)
        If max probability > threshold: prefer Core LDA
        If max probability < threshold: prefer HybridLDA
    
    ensemble_weight : float
        Weight for ensemble (0=use only selected model, 1=equal weighting)
        Default: 0.5 (balanced ensemble)
    
    move_threshold : float
        Threshold for HybridLDA Stage B (default: 0.6)
    
    reg : float
        Regularization for both models (default: 1e-3)
    
    shrinkage_alpha : float or None
        Shrinkage parameter for both models
    
    uc_mu : float
        Update coefficient for mean adaptation (default: 0.4 * 2**-7)
        Lower value = more stable, slower adaptation
    
    use_adaptive_lr : bool
        If True, adaptively adjust uc_mu based on prediction confidence
        Lower confidence -> higher learning rate (default: True)
    
    use_improved_composition : bool
        Use improved probability composition in HybridLDA (default: True)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        ensemble_weight: float = 0.5,
        move_threshold: float = 0.6,
        reg: float = 1e-3,
        shrinkage_alpha: Optional[float] = 0.05,
        uc_mu: float = 0.4 * 2**-7,
        use_adaptive_lr: bool = True,
        use_improved_composition: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.ensemble_weight = ensemble_weight
        self.move_threshold = move_threshold
        self.reg = reg
        self.shrinkage_alpha = shrinkage_alpha
        self.uc_mu = uc_mu
        self.use_adaptive_lr = use_adaptive_lr
        self.use_improved_composition = use_improved_composition
        
        # Models
        self.hybrid_lda: Optional[HybridLDA] = None
        self.core_lda: Optional[LDACore] = None
        
        # Metadata
        self.n_features_ = None
        self.is_fitted_ = False
        self.classes_ = None
        
        # Statistics
        self.n_predictions_ = 0
        self.n_hybrid_selected_ = 0
        self.n_core_selected_ = 0
        self.n_ensemble_used_ = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit both HybridLDA and Core LDA models.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,)
            Class labels (0=rest, 1=left, 2=right)
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")
        
        if len(y) != X.shape[0]:
            raise ValueError(f"X and y must have same length: {X.shape[0]} vs {len(y)}")
        
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Fit HybridLDA
        self.hybrid_lda = HybridLDA(
            move_threshold=self.move_threshold,
            reg=self.reg,
            shrinkage_alpha=self.shrinkage_alpha,
            uc_mu=self.uc_mu,
            use_improved_composition=self.use_improved_composition
        )
        self.hybrid_lda.fit(X, y)
        
        # Fit Core LDA
        self.core_lda = LDACore()
        self.core_lda.fit(X, y)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using adaptive model selection.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features) or (n_features,)
            Feature matrix
        
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted class labels
        """
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return np.array([self.classes_[i] for i in class_indices])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using adaptive ensemble.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features) or (n_features,)
            Feature matrix
        
        Returns:
        --------
        probabilities : np.ndarray, shape (n_samples, n_classes)
            Class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Handle single sample
        single = False
        if X.ndim == 1:
            X = X[np.newaxis, :]
            single = True
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Get predictions from both models
        hybrid_probs = self.hybrid_lda.predict_proba(X)
        core_probs = self.core_lda.predict_proba(X)
        
        # Ensure probabilities are in same class order
        # Both should already be in [0, 1, 2] order, but verify
        assert hybrid_probs.shape[1] == n_classes
        assert core_probs.shape[1] == n_classes
        
        # Adaptive ensemble: weight based on confidence
        final_probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            # Compute confidence (max probability)
            hybrid_conf = np.max(hybrid_probs[i])
            core_conf = np.max(core_probs[i])
            
            # Model selection based on confidence
            if hybrid_conf >= self.confidence_threshold and core_conf >= self.confidence_threshold:
                # Both confident: use ensemble
                weight_hybrid = self.ensemble_weight
                weight_core = 1.0 - self.ensemble_weight
                self.n_ensemble_used_ += 1
            elif core_conf >= self.confidence_threshold:
                # Core LDA is confident: prefer it
                weight_hybrid = 0.2  # Small weight to hybrid
                weight_core = 0.8
                self.n_core_selected_ += 1
            else:
                # Uncertain: prefer HybridLDA (more robust)
                weight_hybrid = 0.8
                weight_core = 0.2
                self.n_hybrid_selected_ += 1
            
            # Normalize weights
            total_weight = weight_hybrid + weight_core
            weight_hybrid /= total_weight
            weight_core /= total_weight
            
            # Ensemble prediction
            final_probs[i] = weight_hybrid * hybrid_probs[i] + weight_core * core_probs[i]
        
        self.n_predictions_ += n_samples
        
        return final_probs[0:1] if single else final_probs
    
    def update(self, label: int, x_feature: np.ndarray):
        """
        Update both models with a new labeled sample (online adaptation).
        
        Uses adaptive learning rate: lower confidence -> higher learning rate.
        
        Parameters:
        -----------
        label : int
            True class label: 0 (rest), 1 (left), or 2 (right)
        x_feature : np.ndarray
            Shape (n_features,) - feature vector for this sample
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        x_feature = np.asarray(x_feature).flatten()
        if len(x_feature) != self.n_features_:
            raise ValueError(f"Feature dimension mismatch: {len(x_feature)} vs {self.n_features_}")
        
        # Get current prediction confidence
        probs = self.predict_proba(x_feature)
        max_prob = np.max(probs[0])
        confidence = max_prob
        
        # Adaptive learning rate: lower confidence -> higher learning rate
        if self.use_adaptive_lr:
            # Inverse relationship: confidence 0.5 -> 2x learning rate, confidence 1.0 -> 1x
            adaptive_uc_mu = self.uc_mu * (2.0 - confidence)
        else:
            adaptive_uc_mu = self.uc_mu
        
        # Temporarily update uc_mu for HybridLDA
        old_uc_mu = self.hybrid_lda.uc_mu
        self.hybrid_lda.uc_mu = adaptive_uc_mu
        
        # Update HybridLDA
        self.hybrid_lda.update(label, x_feature)
        
        # Restore original uc_mu
        self.hybrid_lda.uc_mu = old_uc_mu
        
        # Update Core LDA means (manual EMA update)
        # Core LDA doesn't have update() method, so we update manually
        if label in self.core_lda.mu:
            self.core_lda.mu[label] = (
                (1 - adaptive_uc_mu) * self.core_lda.mu[label] +
                adaptive_uc_mu * x_feature
            )
            # Recompute LDA parameters (w and b)
            self.core_lda._update_parameters()
        
        return self
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        if not self.is_fitted_:
            return {"fitted": False}
        
        stats = {
            "fitted": True,
            "n_features": self.n_features_,
            "n_predictions": self.n_predictions_,
            "n_hybrid_selected": self.n_hybrid_selected_,
            "n_core_selected": self.n_core_selected_,
            "n_ensemble_used": self.n_ensemble_used_,
            "hybrid_selection_rate": (
                self.n_hybrid_selected_ / max(1, self.n_predictions_)
            ),
            "core_selection_rate": (
                self.n_core_selected_ / max(1, self.n_predictions_)
            ),
            "ensemble_rate": (
                self.n_ensemble_used_ / max(1, self.n_predictions_)
            ),
        }
        
        # Add HybridLDA stats
        if self.hybrid_lda is not None:
            stats["hybrid_lda"] = self.hybrid_lda.get_update_stats()
        
        return stats
    
    def get_stage_info(self) -> Dict[str, Any]:
        """Get information about the fitted models."""
        if not self.is_fitted_:
            return {"fitted": False}
        
        info = {
            "fitted": True,
            "n_features": self.n_features_,
            "classes": self.classes_.tolist(),
            "confidence_threshold": self.confidence_threshold,
            "ensemble_weight": self.ensemble_weight,
            "use_adaptive_lr": self.use_adaptive_lr,
        }
        
        if self.hybrid_lda is not None:
            info["hybrid_lda"] = self.hybrid_lda.get_stage_info()
        
        return info

"""
Enhanced Adaptive LDA with Multiple Improvements.

This module extends HybridLDA with:
1. Adaptive Model Selection (Standard LDA + HybridLDA)
2. Improved Probability Composition (confidence-weighted Stage B)
3. Temporal Smoothing (majority/weighted voting)
4. Dynamic Threshold Adaptation
5. Better Adaptation Strategy (adaptive learning rate)
"""

import numpy as np
from collections import deque
from .hybrid_lda import HybridLDA
from .lda_core import fit_lda, predict_proba_lda, softmax


class EnhancedAdaptiveLDA:
    """
    Enhanced Adaptive LDA with multiple improvements for better accuracy.
    
    Improvements:
    1. Adaptive Model Selection: Uses Standard LDA when confident, HybridLDA when uncertain
    2. Improved Probability Composition: Confidence-weighted Stage B probabilities
    3. Temporal Smoothing: Majority/weighted voting over recent predictions
    4. Dynamic Threshold Adaptation: Adjusts move_threshold based on recent performance
    5. Adaptive Learning Rate: Adjusts uc_mu based on prediction correctness and confidence
    """
    
    def __init__(
        self,
        move_threshold=0.5,
        always_run_stage_b=True,
        reg=1e-2,
        shrinkage_alpha=0.1,
        uc_mu=0.4 * 2**-6,
        # Enhanced features
        use_adaptive_selection=True,
        use_improved_composition=True,
        use_temporal_smoothing=True,
        use_dynamic_threshold=True,
        use_adaptive_learning_rate=True,
        # Temporal smoothing parameters
        smoothing_window_size=5,
        smoothing_method='weighted',  # 'majority', 'weighted', 'exponential'
        # Dynamic threshold parameters
        threshold_adaptation_rate=0.05,
        min_threshold=0.3,
        max_threshold=0.8,
        # Adaptive learning rate parameters
        learning_rate_multipliers={
            'wrong_low_conf': 3.0,    # Wrong prediction + low confidence
            'wrong_high_conf': 2.0,    # Wrong prediction + high confidence
            'correct_low_conf': 1.5,   # Correct but uncertain
            'correct_high_conf': 0.5   # Correct and confident
        },
        # Adaptive selection parameters
        selection_confidence_threshold=0.7,
    ):
        """
        Initialize Enhanced Adaptive LDA.
        
        Parameters:
        -----------
        move_threshold : float
            Initial confidence threshold for triggering Stage B
        always_run_stage_b : bool
            If True, always compute Stage B probabilities
        reg : float
            Regularization coefficient for covariance
        shrinkage_alpha : float or None
            Shrinkage parameter in [0, 1]
        uc_mu : float
            Base update coefficient for mean adaptation
        use_adaptive_selection : bool
            Enable adaptive model selection (Standard LDA + HybridLDA)
        use_improved_composition : bool
            Enable confidence-weighted probability composition
        use_temporal_smoothing : bool
            Enable temporal smoothing over recent predictions
        use_dynamic_threshold : bool
            Enable dynamic threshold adaptation
        use_adaptive_learning_rate : bool
            Enable adaptive learning rate based on prediction quality
        smoothing_window_size : int
            Number of recent predictions to consider for smoothing
        smoothing_method : str
            Smoothing method: 'majority', 'weighted', or 'exponential'
        threshold_adaptation_rate : float
            Step size for threshold adjustments
        min_threshold : float
            Minimum allowed move_threshold
        max_threshold : float
            Maximum allowed move_threshold
        learning_rate_multipliers : dict
            Multipliers for adaptive learning rate
        selection_confidence_threshold : float
            Confidence threshold for adaptive model selection
        """
        # Core HybridLDA model
        # Note: use_improved_composition=False because EnhancedAdaptiveLDA has its own improved composition logic
        self.hybrid_lda = HybridLDA(
            move_threshold=move_threshold,
            always_run_stage_b=always_run_stage_b,
            reg=reg,
            shrinkage_alpha=shrinkage_alpha,
            uc_mu=uc_mu,
            use_improved_composition=False  # EnhancedAdaptiveLDA applies its own improved composition
        )
        
        # Standard 3-class LDA for adaptive selection
        self.standard_lda_means_ = None
        self.standard_lda_cov_ = None
        self.standard_lda_inv_cov_ = None
        self.standard_lda_classes_ = None
        self.standard_lda_priors_ = None
        
        # Feature flags
        self.use_adaptive_selection = use_adaptive_selection
        self.use_improved_composition = use_improved_composition
        self.use_temporal_smoothing = use_temporal_smoothing
        self.use_dynamic_threshold = use_dynamic_threshold
        self.use_adaptive_learning_rate = use_adaptive_learning_rate
        
        # Temporal smoothing buffers
        self.smoothing_window_size = smoothing_window_size
        self.smoothing_method = smoothing_method
        self.prediction_buffer = deque(maxlen=smoothing_window_size)
        self.probability_buffer = deque(maxlen=smoothing_window_size)
        self.confidence_buffer = deque(maxlen=smoothing_window_size)
        
        # Dynamic threshold tracking
        self.threshold_adaptation_rate = threshold_adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.recent_predictions = deque(maxlen=50)
        self.recent_labels = deque(maxlen=50)
        self.recent_confidences = deque(maxlen=50)
        
        # Adaptive learning rate
        self.base_uc_mu = uc_mu
        self.learning_rate_multipliers = learning_rate_multipliers
        
        # Adaptive selection
        self.selection_confidence_threshold = selection_confidence_threshold
        
        # Metadata
        self.n_features_ = None
        self.is_fitted_ = False
    
    def fit(self, features, labels):
        """
        Fit both HybridLDA and Standard LDA models.
        
        Parameters:
        -----------
        features : np.ndarray
            Shape (n_samples, n_features) - extracted features
        labels : np.ndarray
            Shape (n_samples,) - class labels (0=rest, 1=left, 2=right)
        """
        # Fit HybridLDA
        self.hybrid_lda.fit(features, labels)
        
        # Fit Standard 3-class LDA if adaptive selection is enabled
        if self.use_adaptive_selection:
            (self.standard_lda_means_,
             self.standard_lda_cov_,
             self.standard_lda_inv_cov_,
             self.standard_lda_classes_,
             self.standard_lda_priors_) = fit_lda(
                features, labels,
                reg=self.hybrid_lda.reg,
                shrinkage_alpha=self.hybrid_lda.shrinkage_alpha,
                compute_priors=True
            )
        
        self.n_features_ = features.shape[1]
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, features):
        """
        Predict 3-class probabilities with enhanced features.
        
        Parameters:
        -----------
        features : np.ndarray
            Shape (n_samples, n_features) or (n_features,)
        
        Returns:
        --------
        probabilities : np.ndarray
            Shape (n_samples, 3) - [p_rest, p_left, p_right]
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Handle single sample
        single_sample = False
        if features.ndim == 1:
            features = features[np.newaxis, :]
            single_sample = True
        
        n_samples = features.shape[0]
        probs_3class = np.zeros((n_samples, 3))
        
        # Get HybridLDA probabilities
        hybrid_probs = self.hybrid_lda.predict_proba(features)
        
        # Adaptive Model Selection
        if self.use_adaptive_selection:
            # Get Standard LDA probabilities
            standard_probs = predict_proba_lda(
                features,
                self.standard_lda_means_,
                self.standard_lda_inv_cov_,
                self.standard_lda_classes_,
                self.standard_lda_priors_,
                method='discriminant'
            )
            
            # Compute confidences
            hybrid_confidence = hybrid_probs.max(axis=1)
            standard_confidence = standard_probs.max(axis=1)
            
            # Choose model based on confidence
            use_standard = standard_confidence >= self.selection_confidence_threshold
            use_hybrid = ~use_standard
            
            # Combine probabilities
            probs_3class[use_standard] = standard_probs[use_standard]
            probs_3class[use_hybrid] = hybrid_probs[use_hybrid]
        else:
            probs_3class = hybrid_probs
        
        # Improved Probability Composition (if not using adaptive selection)
        if self.use_improved_composition and not self.use_adaptive_selection:
            probs_3class = self._improved_composition(features, probs_3class)
        
        # Normalize
        row_sums = probs_3class.sum(axis=1, keepdims=True)
        probs_3class = probs_3class / (row_sums + 1e-10)
        
        return probs_3class[0:1] if single_sample else probs_3class
    
    def _improved_composition(self, features, probs_3class):
        """
        Improved probability composition with confidence weighting.
        
        Uses Stage B confidence to weight the composition.
        """
        # Get Stage A and Stage B probabilities
        probs_a = predict_proba_lda(
            features,
            self.hybrid_lda.stage_a_means_,
            self.hybrid_lda.stage_a_inv_cov_,
            self.hybrid_lda.stage_a_classes_,
            self.hybrid_lda.stage_a_priors_,
            method='discriminant'
        )
        p_rest = probs_a[:, 0]
        p_move = probs_a[:, 1]
        
        probs_b = predict_proba_lda(
            features,
            self.hybrid_lda.stage_b_means_,
            self.hybrid_lda.stage_b_inv_cov_,
            self.hybrid_lda.stage_b_classes_,
            self.hybrid_lda.stage_b_priors_,
            method='discriminant'
        )
        p_left_given_move = probs_b[:, 0]
        p_right_given_move = probs_b[:, 1]
        
        # Compute Stage B confidence
        stage_b_confidence = np.maximum(p_left_given_move, p_right_given_move)
        stage_b_weight = np.clip(stage_b_confidence * 2 - 1, 0, 1)
        
        # Improved composition with confidence weighting
        probs_3class[:, 0] = p_rest
        probs_3class[:, 1] = p_move * (
            stage_b_weight * p_left_given_move +
            (1 - stage_b_weight) * 0.5
        )
        probs_3class[:, 2] = p_move * (
            stage_b_weight * p_right_given_move +
            (1 - stage_b_weight) * 0.5
        )
        
        return probs_3class
    
    def predict(self, features):
        """
        Predict class labels with temporal smoothing.
        
        Parameters:
        -----------
        features : np.ndarray
            Shape (n_samples, n_features) or (n_features,)
        
        Returns:
        --------
        predictions : np.ndarray
            Shape (n_samples,) or scalar
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Handle single sample
        single_sample = False
        if features.ndim == 1:
            features = features[np.newaxis, :]
            single_sample = True
        
        # Get probabilities
        probs = self.predict_proba(features)
        
        # Get Stage A movement probability for gating
        probs_a = predict_proba_lda(
            features,
            self.hybrid_lda.stage_a_means_,
            self.hybrid_lda.stage_a_inv_cov_,
            self.hybrid_lda.stage_a_classes_,
            self.hybrid_lda.stage_a_priors_,
            method='discriminant'
        )
        p_move = probs_a[:, 1]
        
        # Predict with gating
        predictions = np.zeros(len(features), dtype=int)
        confidences = np.zeros(len(features))
        
        for i in range(len(features)):
            if p_move[i] < self.hybrid_lda.move_threshold:
                predictions[i] = 0  # Rest
            else:
                predictions[i] = np.argmax(probs[i])
            confidences[i] = probs[i].max()
        
        # Temporal smoothing
        if self.use_temporal_smoothing:
            predictions = self._apply_temporal_smoothing(predictions, confidences)
        
        return predictions[0] if single_sample else predictions
    
    def _apply_temporal_smoothing(self, predictions, confidences):
        """
        Apply temporal smoothing to predictions.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Current predictions
        confidences : np.ndarray
            Current prediction confidences
        
        Returns:
        --------
        smoothed_predictions : np.ndarray
            Smoothed predictions
        """
        smoothed = np.zeros_like(predictions)
        
        for i in range(len(predictions)):
            # Add current prediction to buffer
            self.prediction_buffer.append(predictions[i])
            self.confidence_buffer.append(confidences[i])
            
            if len(self.prediction_buffer) < 2:
                # Not enough history, use current prediction
                smoothed[i] = predictions[i]
            else:
                # Apply smoothing
                if self.smoothing_method == 'majority':
                    # Simple majority vote
                    smoothed[i] = np.bincount(list(self.prediction_buffer)).argmax()
                
                elif self.smoothing_method == 'weighted':
                    # Weighted vote by confidence
                    weights = np.array(list(self.confidence_buffer))
                    preds = np.array(list(self.prediction_buffer))
                    smoothed[i] = np.bincount(preds, weights=weights).argmax()
                
                elif self.smoothing_method == 'exponential':
                    # Exponential decay (recent = higher weight)
                    n = len(self.prediction_buffer)
                    exp_weights = np.exp(np.linspace(-2, 0, n))
                    preds = np.array(list(self.prediction_buffer))
                    smoothed[i] = np.bincount(preds, weights=exp_weights).argmax()
                else:
                    smoothed[i] = predictions[i]
        
        return smoothed
    
    def update(self, label, x_feature):
        """
        Update classifier with adaptive learning rate.
        
        Parameters:
        -----------
        label : int
            True class label: 0 (rest), 1 (left), or 2 (right)
        x_feature : np.ndarray
            Shape (n_features,) - feature vector for this sample
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get prediction before update (for adaptive learning rate and dynamic threshold)
        prediction = None
        prediction_confidence = None
        
        if self.use_adaptive_learning_rate or self.use_dynamic_threshold:
            prediction = self.predict(x_feature)
            probs = self.predict_proba(x_feature)
            prediction_confidence = probs.max()
        
        # Adaptive learning rate logic
        if self.use_adaptive_learning_rate:
            # Determine adaptive learning rate
            is_correct = (prediction == label)
            
            if not is_correct and prediction_confidence < 0.5:
                # Wrong prediction + low confidence: adapt aggressively
                multiplier = self.learning_rate_multipliers['wrong_low_conf']
            elif not is_correct and prediction_confidence >= 0.5:
                # Wrong prediction but was confident: adapt moderately
                multiplier = self.learning_rate_multipliers['wrong_high_conf']
            elif is_correct and prediction_confidence < 0.5:
                # Correct but uncertain: adapt moderately
                multiplier = self.learning_rate_multipliers['correct_low_conf']
            else:
                # Correct and confident: adapt slowly
                multiplier = self.learning_rate_multipliers['correct_high_conf']
            
            # Temporarily adjust uc_mu
            old_uc_mu = self.hybrid_lda.uc_mu
            self.hybrid_lda.uc_mu = self.base_uc_mu * multiplier
        
        # Update HybridLDA
        self.hybrid_lda.update(label, x_feature)
        
        # Restore uc_mu if adaptive learning rate was used
        if self.use_adaptive_learning_rate:
            self.hybrid_lda.uc_mu = old_uc_mu
        
        # Track for dynamic threshold adaptation
        if self.use_dynamic_threshold:
            self.recent_predictions.append(prediction)
            self.recent_labels.append(label)
            self.recent_confidences.append(prediction_confidence)
            
            # Update threshold periodically
            if len(self.recent_predictions) >= 10 and len(self.recent_predictions) % 10 == 0:
                self._update_threshold()
        
        return self
    
    def _update_threshold(self):
        """Update move_threshold based on recent performance."""
        if len(self.recent_predictions) < 10:
            return
        
        # Compute recent accuracy
        recent_accuracy = np.mean(
            np.array(self.recent_predictions) == np.array(self.recent_labels)
        )
        avg_confidence = np.mean(list(self.recent_confidences))
        
        # Adjust threshold
        if recent_accuracy < 0.4:
            # Low accuracy: be more conservative
            new_threshold = min(
                self.max_threshold,
                self.hybrid_lda.move_threshold + self.threshold_adaptation_rate
            )
        elif recent_accuracy > 0.6:
            # High accuracy: can be more aggressive
            new_threshold = max(
                self.min_threshold,
                self.hybrid_lda.move_threshold - self.threshold_adaptation_rate
            )
        else:
            # Moderate accuracy: adjust based on confidence
            if avg_confidence < 0.5:
                new_threshold = min(
                    self.max_threshold,
                    self.hybrid_lda.move_threshold + 0.02
                )
            else:
                new_threshold = self.hybrid_lda.move_threshold
        
        self.hybrid_lda.move_threshold = new_threshold
    
    def get_update_stats(self):
        """Get online adaptation statistics."""
        stats = self.hybrid_lda.get_update_stats()
        stats['current_threshold'] = self.hybrid_lda.move_threshold
        stats['recent_accuracy'] = (
            np.mean(np.array(self.recent_predictions) == np.array(self.recent_labels))
            if len(self.recent_predictions) > 0 else None
        )
        return stats
    
    def get_stage_info(self):
        """Get information about the fitted stages."""
        info = self.hybrid_lda.get_stage_info()
        info['enhanced_features'] = {
            'adaptive_selection': self.use_adaptive_selection,
            'improved_composition': self.use_improved_composition,
            'temporal_smoothing': self.use_temporal_smoothing,
            'dynamic_threshold': self.use_dynamic_threshold,
            'adaptive_learning_rate': self.use_adaptive_learning_rate,
        }
        return info

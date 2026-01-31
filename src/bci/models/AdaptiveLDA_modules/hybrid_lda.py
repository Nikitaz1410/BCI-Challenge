"""
Hybrid 2-Stage LDA Classifier for Motor Imagery BCI.

Implements a hierarchical classification approach:
- Stage A: Binary LDA for Rest vs Movement (Left OR Right)
- Stage B: Binary LDA for Left vs Right (only if Movement is confident)

This approach often works better than 3-class LDA because:
1. Rest class often overlaps with both Left and Right
2. Binary classifiers are more stable
3. Allows confidence gating to reduce false positives

Probability composition:
    p_rest  = p_A(rest)
    p_move  = p_A(move)
    p_left  = p_move * p_B(left|move)
    p_right = p_move * p_B(right|move)
"""

import numpy as np
from .lda_core import (
    fit_lda,
    predict_lda,
    predict_proba_lda,
    compute_discriminant_scores,
    softmax
)


class HybridLDA:
    """
    2-Stage Hybrid LDA Classifier.

    Stage A: Rest (0) vs Movement (1+2 combined)
    Stage B: Left (1) vs Right (2)

    Parameters:
    -----------
    move_threshold : float
        Confidence threshold for triggering Stage B (default: 0.6)
        If p_A(movement) < threshold, predict Rest without running Stage B.
    always_run_stage_b : bool
        If True, always run Stage B but gate output based on threshold.
        If False, skip Stage B entirely when movement probability is low.
    reg : float
        Regularization coefficient for covariance (default: 1e-2)
    shrinkage_alpha : float or None
        Shrinkage parameter in [0, 1]. If None, uses reg instead.
    """

    def __init__(self, move_threshold=0.6, always_run_stage_b=True,
                 reg=1e-2, shrinkage_alpha=None, uc_mu=0.4 * 2**-6):
        """
        Initialize Hybrid 2-Stage LDA.

        Parameters:
        -----------
        move_threshold : float
            Confidence threshold for triggering Stage B (default: 0.6)
        always_run_stage_b : bool
            If True, always compute Stage B probabilities
        reg : float
            Regularization coefficient for covariance
        shrinkage_alpha : float or None
            Shrinkage parameter in [0, 1]
        uc_mu : float
            Update coefficient for mean adaptation (default: 0.00625 from Wu et al.)
            Used in online update: mu_new = (1 - uc_mu) * mu_old + uc_mu * x
        """
        self.move_threshold = move_threshold
        self.always_run_stage_b = always_run_stage_b
        self.reg = reg
        self.shrinkage_alpha = shrinkage_alpha
        self.uc_mu = uc_mu

        # Stage A: Rest vs Movement
        self.stage_a_means_ = None
        self.stage_a_cov_ = None
        self.stage_a_inv_cov_ = None
        self.stage_a_classes_ = None  # [0, 1] where 1 = movement
        self.stage_a_priors_ = None

        # Stage B: Left vs Right
        self.stage_b_means_ = None
        self.stage_b_cov_ = None
        self.stage_b_inv_cov_ = None
        self.stage_b_classes_ = None  # [1, 2] for left, right
        self.stage_b_priors_ = None

        # Metadata
        self.n_features_ = None
        self.is_fitted_ = False

        # Update counters for debugging
        self.n_updates_ = 0
        self.n_updates_stage_a_ = {'rest': 0, 'move': 0}
        self.n_updates_stage_b_ = {'left': 0, 'right': 0}

    def fit(self, features, labels):
        """
        Fit the 2-stage hybrid LDA.

        Parameters:
        -----------
        features : np.ndarray
            Shape (n_samples, n_features) - extracted features
        labels : np.ndarray
            Shape (n_samples,) - class labels (0=rest, 1=left, 2=right)

        Returns:
        --------
        self
        """
        # Validate inputs
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape {labels.shape}")
        if len(features) != len(labels):
            raise ValueError("features and labels must have same length")

        unique_labels = np.unique(labels)
        if not all(l in unique_labels for l in [0, 1, 2]):
            raise ValueError(f"Expected labels [0, 1, 2], got {unique_labels}")

        self.n_features_ = features.shape[1]

        # =================================================================
        # STAGE A: Rest (0) vs Movement (1 or 2)
        # =================================================================
        # Create binary labels: 0 = rest, 1 = movement
        labels_a = np.where(labels == 0, 0, 1)

        (self.stage_a_means_,
         self.stage_a_cov_,
         self.stage_a_inv_cov_,
         self.stage_a_classes_,
         self.stage_a_priors_) = fit_lda(
            features, labels_a,
            reg=self.reg,
            shrinkage_alpha=self.shrinkage_alpha,
            compute_priors=True
        )

        # =================================================================
        # STAGE B: Left (1) vs Right (2) - only on movement samples
        # =================================================================
        movement_mask = labels != 0
        features_move = features[movement_mask]
        labels_move = labels[movement_mask]

        if len(np.unique(labels_move)) < 2:
            raise ValueError("Need both Left and Right samples for Stage B")

        (self.stage_b_means_,
         self.stage_b_cov_,
         self.stage_b_inv_cov_,
         self.stage_b_classes_,
         self.stage_b_priors_) = fit_lda(
            features_move, labels_move,
            reg=self.reg,
            shrinkage_alpha=self.shrinkage_alpha,
            compute_priors=True
        )

        self.is_fitted_ = True
        return self

    def predict_proba(self, features):
        """
        Predict 3-class probabilities using hierarchical composition.

        p_rest  = p_A(rest)
        p_left  = p_A(move) * p_B(left|move)
        p_right = p_A(move) * p_B(right|move)

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

        # =================================================================
        # Stage A: Get p(rest) and p(movement)
        # =================================================================
        probs_a = predict_proba_lda(
            features,
            self.stage_a_means_,
            self.stage_a_inv_cov_,
            self.stage_a_classes_,
            self.stage_a_priors_,
            method='discriminant'
        )
        # probs_a[:, 0] = p(rest), probs_a[:, 1] = p(movement)
        p_rest = probs_a[:, 0]
        p_move = probs_a[:, 1]

        # =================================================================
        # Stage B: Get p(left|move) and p(right|move)
        # =================================================================
        if self.always_run_stage_b:
            # Always compute Stage B probabilities
            probs_b = predict_proba_lda(
                features,
                self.stage_b_means_,
                self.stage_b_inv_cov_,
                self.stage_b_classes_,
                self.stage_b_priors_,
                method='discriminant'
            )
            # stage_b_classes_ is [1, 2] so:
            # probs_b[:, 0] = p(left|move), probs_b[:, 1] = p(right|move)
            p_left_given_move = probs_b[:, 0]
            p_right_given_move = probs_b[:, 1]

            # Compose probabilities
            probs_3class[:, 0] = p_rest  # Rest
            probs_3class[:, 1] = p_move * p_left_given_move  # Left
            probs_3class[:, 2] = p_move * p_right_given_move  # Right
        else:
            # Only run Stage B if movement is confident
            for i in range(n_samples):
                if p_move[i] >= self.move_threshold:
                    # Run Stage B for this sample
                    probs_b_i = predict_proba_lda(
                        features[i:i+1],
                        self.stage_b_means_,
                        self.stage_b_inv_cov_,
                        self.stage_b_classes_,
                        self.stage_b_priors_,
                        method='discriminant'
                    )
                    p_left_given_move = probs_b_i[0, 0]
                    p_right_given_move = probs_b_i[0, 1]

                    probs_3class[i, 0] = p_rest[i]
                    probs_3class[i, 1] = p_move[i] * p_left_given_move
                    probs_3class[i, 2] = p_move[i] * p_right_given_move
                else:
                    # Skip Stage B, assign all movement probability equally
                    probs_3class[i, 0] = p_rest[i]
                    probs_3class[i, 1] = p_move[i] * 0.5
                    probs_3class[i, 2] = p_move[i] * 0.5

        # Normalize to ensure sum = 1 (should already be close)
        row_sums = probs_3class.sum(axis=1, keepdims=True)
        probs_3class = probs_3class / (row_sums + 1e-10)

        return probs_3class[0:1] if single_sample else probs_3class

    def predict(self, features):
        """
        Predict class labels (0=rest, 1=left, 2=right).

        Uses confidence gating:
        - If p(movement) < threshold, predict Rest
        - Otherwise, predict argmax of 3-class probabilities

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

        # Get 3-class probabilities
        probs = self.predict_proba(features)

        # Get Stage A movement probability for gating
        probs_a = predict_proba_lda(
            features,
            self.stage_a_means_,
            self.stage_a_inv_cov_,
            self.stage_a_classes_,
            self.stage_a_priors_,
            method='discriminant'
        )
        p_move = probs_a[:, 1]

        # Predict with gating
        predictions = np.zeros(len(features), dtype=int)
        for i in range(len(features)):
            if p_move[i] < self.move_threshold:
                # Not confident in movement, predict Rest
                predictions[i] = 0
            else:
                # Confident in movement, use 3-class argmax
                predictions[i] = np.argmax(probs[i])

        return predictions[0] if single_sample else predictions

    def predict_with_details(self, features):
        """
        Predict with full diagnostic information.

        Returns:
        --------
        results : dict with keys:
            - 'predictions': final 3-class predictions
            - 'probs_3class': 3-class probabilities
            - 'p_rest': Stage A rest probability
            - 'p_move': Stage A movement probability
            - 'p_left_given_move': Stage B left|movement probability
            - 'p_right_given_move': Stage B right|movement probability
            - 'stage_b_triggered': bool array, whether Stage B was decisive
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        if features.ndim == 1:
            features = features[np.newaxis, :]

        # Stage A
        probs_a = predict_proba_lda(
            features,
            self.stage_a_means_,
            self.stage_a_inv_cov_,
            self.stage_a_classes_,
            self.stage_a_priors_,
            method='discriminant'
        )
        p_rest = probs_a[:, 0]
        p_move = probs_a[:, 1]

        # Stage B
        probs_b = predict_proba_lda(
            features,
            self.stage_b_means_,
            self.stage_b_inv_cov_,
            self.stage_b_classes_,
            self.stage_b_priors_,
            method='discriminant'
        )
        p_left_given_move = probs_b[:, 0]
        p_right_given_move = probs_b[:, 1]

        # 3-class probabilities
        probs_3class = self.predict_proba(features)

        # Predictions
        predictions = self.predict(features)

        # Stage B triggered?
        stage_b_triggered = p_move >= self.move_threshold

        return {
            'predictions': predictions,
            'probs_3class': probs_3class,
            'p_rest': p_rest,
            'p_move': p_move,
            'p_left_given_move': p_left_given_move,
            'p_right_given_move': p_right_given_move,
            'stage_b_triggered': stage_b_triggered
        }

    def update(self, label, x_feature):
        """
        Update classifier means with a new labeled sample (online adaptation).

        Uses exponential moving average (EMA) for mean-only adaptation:
            mu_new = (1 - uc_mu) * mu_old + uc_mu * x

        Covariance is kept FIXED for online stability (as recommended by Wu et al.).

        Parameters:
        -----------
        label : int
            True class label: 0 (rest), 1 (left), or 2 (right)
        x_feature : np.ndarray
            Shape (n_features,) - feature vector for this sample

        Returns:
        --------
        self : HybridLDA
            Returns self for method chaining

        Notes:
        ------
        - Stage A is ALWAYS updated (rest vs movement)
        - Stage B is ONLY updated when label in {1, 2} (movement classes)
        - Uses mean-only adaptation for stability
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        # Validate inputs
        if label not in [0, 1, 2]:
            raise ValueError(f"label must be 0, 1, or 2, got {label}")

        x_feature = np.asarray(x_feature).flatten()
        if len(x_feature) != self.n_features_:
            raise ValueError(f"Feature dimension mismatch: {len(x_feature)} vs {self.n_features_}")

        if not np.all(np.isfinite(x_feature)):
            raise ValueError("x_feature contains non-finite values")

        # =================================================================
        # STAGE A UPDATE: Rest (0) vs Movement (1 or 2)
        # =================================================================
        # Binary label for Stage A: 0 = rest, 1 = movement
        label_a = 0 if label == 0 else 1

        # stage_a_means_ is a dict with keys [0, 1], use label_a directly as key
        # EMA update for Stage A mean
        self._update_mean(self.stage_a_means_, label_a, x_feature)

        # Update counter
        if label_a == 0:
            self.n_updates_stage_a_['rest'] += 1
        else:
            self.n_updates_stage_a_['move'] += 1

        # =================================================================
        # STAGE B UPDATE: Left (1) vs Right (2) - only for movement
        # =================================================================
        if label in [1, 2]:
            # stage_b_means_ is a dict with keys [1, 2], use label directly as key
            # EMA update for Stage B mean
            self._update_mean(self.stage_b_means_, label, x_feature)

            # Update counter
            if label == 1:
                self.n_updates_stage_b_['left'] += 1
            else:
                self.n_updates_stage_b_['right'] += 1

        self.n_updates_ += 1
        return self

    def _update_mean(self, means_dict, class_label, x_feature):
        """
        Apply EMA update to a class mean (in-place).

        mu_new = (1 - uc_mu) * mu_old + uc_mu * x
        
        Parameters:
        -----------
        means_dict : dict
            Dictionary mapping class labels to mean vectors
        class_label : int
            Class label (used as dictionary key)
        x_feature : np.ndarray
            New feature vector
        """
        means_dict[class_label] = (
            (1 - self.uc_mu) * means_dict[class_label] +
            self.uc_mu * x_feature
        )

    def get_update_stats(self):
        """Get online adaptation statistics."""
        return {
            'n_updates': self.n_updates_,
            'stage_a': self.n_updates_stage_a_.copy(),
            'stage_b': self.n_updates_stage_b_.copy(),
            'uc_mu': self.uc_mu
        }

    def get_stage_info(self):
        """Get information about the fitted stages."""
        if not self.is_fitted_:
            return {"fitted": False}

        return {
            "fitted": True,
            "n_features": self.n_features_,
            "stage_a": {
                "classes": self.stage_a_classes_.tolist(),
                "priors": self.stage_a_priors_.tolist(),
                "description": "Rest (0) vs Movement (1)"
            },
            "stage_b": {
                "classes": self.stage_b_classes_.tolist(),
                "priors": self.stage_b_priors_.tolist(),
                "description": "Left (1) vs Right (2)"
            },
            "move_threshold": self.move_threshold,
            "always_run_stage_b": self.always_run_stage_b,
            "uc_mu": self.uc_mu
        }


# =============================================================================
# INTEGRATION TEST: Online Adaptation
# =============================================================================

def test_hybrid_lda_online_adaptation():
    """
    Integration test that simulates streaming segments and verifies
    that update() changes means and predicted probabilities.

    Run with:
        python -c "from bci.Models.AdaptiveLDA_modules.hybrid_lda import test_hybrid_lda_online_adaptation; test_hybrid_lda_online_adaptation()"
    """
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: HybridLDA Online Adaptation")
    print("=" * 60)

    np.random.seed(42)

    # =================================================================
    # 1. Create synthetic training data
    # =================================================================
    n_features = 20
    n_samples_per_class = 30

    # Create class-specific distributions
    # Rest: centered around [0, 0, ...]
    # Left: centered around [1, 0, ...]
    # Right: centered around [0, 1, ...]
    rest_mean = np.zeros(n_features)
    left_mean = np.zeros(n_features)
    left_mean[0] = 2.0  # Distinguish left
    right_mean = np.zeros(n_features)
    right_mean[1] = 2.0  # Distinguish right

    X_rest = np.random.randn(n_samples_per_class, n_features) + rest_mean
    X_left = np.random.randn(n_samples_per_class, n_features) + left_mean
    X_right = np.random.randn(n_samples_per_class, n_features) + right_mean

    X_train = np.vstack([X_rest, X_left, X_right])
    y_train = np.array([0] * n_samples_per_class +
                       [1] * n_samples_per_class +
                       [2] * n_samples_per_class)

    print(f"\n[1] Training data: {X_train.shape[0]} samples, {n_features} features")

    # =================================================================
    # 2. Fit initial HybridLDA
    # =================================================================
    hybrid = HybridLDA(move_threshold=0.5, uc_mu=0.1)  # Higher uc_mu for visible changes
    hybrid.fit(X_train, y_train)

    print(f"[2] Initial fit complete")
    print(f"    Stage A means shape: {hybrid.stage_a_means_.shape}")
    print(f"    Stage B means shape: {hybrid.stage_b_means_.shape}")

    # Store initial means for comparison
    stage_a_means_before = hybrid.stage_a_means_.copy()
    stage_b_means_before = hybrid.stage_b_means_.copy()

    # =================================================================
    # 3. Get initial predictions on test sample
    # =================================================================
    # Create a test sample that's ambiguous
    test_sample = np.random.randn(n_features) + 0.5 * left_mean
    probs_before = hybrid.predict_proba(test_sample)
    pred_before = hybrid.predict(test_sample)

    print(f"\n[3] Before adaptation:")
    print(f"    Test sample prediction: {pred_before}")
    print(f"    Probabilities: Rest={probs_before[0, 0]:.3f}, Left={probs_before[0, 1]:.3f}, Right={probs_before[0, 2]:.3f}")

    # =================================================================
    # 4. Simulate streaming: Update with new samples
    # =================================================================
    print(f"\n[4] Simulating online streaming...")

    # Simulate 20 streaming segments with "left" feedback
    n_updates = 20
    for i in range(n_updates):
        # New sample comes in, user indicates it was "left"
        new_sample = np.random.randn(n_features) + left_mean * 1.5  # Stronger left signal
        hybrid.update(label=1, x_feature=new_sample)

    stats = hybrid.get_update_stats()
    print(f"    Updates performed: {stats['n_updates']}")
    print(f"    Stage A updates: {stats['stage_a']}")
    print(f"    Stage B updates: {stats['stage_b']}")

    # =================================================================
    # 5. Verify means have changed
    # =================================================================
    stage_a_means_after = hybrid.stage_a_means_.copy()
    stage_b_means_after = hybrid.stage_b_means_.copy()

    # Stage A movement mean should change (index 1)
    stage_a_move_change = np.linalg.norm(stage_a_means_after[1] - stage_a_means_before[1])
    # Stage A rest mean should NOT change (no rest updates)
    stage_a_rest_change = np.linalg.norm(stage_a_means_after[0] - stage_a_means_before[0])

    # Stage B left mean should change (index 0, since classes=[1,2])
    stage_b_left_change = np.linalg.norm(stage_b_means_after[0] - stage_b_means_before[0])
    # Stage B right mean should NOT change (no right updates)
    stage_b_right_change = np.linalg.norm(stage_b_means_after[1] - stage_b_means_before[1])

    print(f"\n[5] Mean changes:")
    print(f"    Stage A rest mean change:  {stage_a_rest_change:.6f} (expected: 0)")
    print(f"    Stage A move mean change:  {stage_a_move_change:.6f} (expected: >0)")
    print(f"    Stage B left mean change:  {stage_b_left_change:.6f} (expected: >0)")
    print(f"    Stage B right mean change: {stage_b_right_change:.6f} (expected: 0)")

    # =================================================================
    # 6. Verify predictions have changed
    # =================================================================
    probs_after = hybrid.predict_proba(test_sample)
    pred_after = hybrid.predict(test_sample)

    prob_change = np.abs(probs_after - probs_before).sum()

    print(f"\n[6] After adaptation:")
    print(f"    Test sample prediction: {pred_after}")
    print(f"    Probabilities: Rest={probs_after[0, 0]:.3f}, Left={probs_after[0, 1]:.3f}, Right={probs_after[0, 2]:.3f}")
    print(f"    Total probability change: {prob_change:.4f}")

    # =================================================================
    # 7. Assertions
    # =================================================================
    print(f"\n[7] Verification:")

    all_passed = True

    # Check 1: Stage A movement mean changed
    if stage_a_move_change > 0.01:
        print(f"    ✓ Stage A movement mean updated (Δ={stage_a_move_change:.4f})")
    else:
        print(f"    ✗ Stage A movement mean NOT updated")
        all_passed = False

    # Check 2: Stage A rest mean unchanged
    if stage_a_rest_change < 1e-10:
        print(f"    ✓ Stage A rest mean unchanged")
    else:
        print(f"    ✗ Stage A rest mean changed unexpectedly")
        all_passed = False

    # Check 3: Stage B left mean changed
    if stage_b_left_change > 0.01:
        print(f"    ✓ Stage B left mean updated (Δ={stage_b_left_change:.4f})")
    else:
        print(f"    ✗ Stage B left mean NOT updated")
        all_passed = False

    # Check 4: Stage B right mean unchanged
    if stage_b_right_change < 1e-10:
        print(f"    ✓ Stage B right mean unchanged")
    else:
        print(f"    ✗ Stage B right mean changed unexpectedly")
        all_passed = False

    # Check 5: Probabilities changed
    if prob_change > 0.001:
        print(f"    ✓ Predicted probabilities changed (Δ={prob_change:.4f})")
    else:
        print(f"    ✗ Predicted probabilities NOT changed")
        all_passed = False

    # Check 6: Update counters correct
    if stats['n_updates'] == n_updates:
        print(f"    ✓ Update counter correct ({stats['n_updates']})")
    else:
        print(f"    ✗ Update counter incorrect")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL INTEGRATION TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED - check above")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    test_hybrid_lda_online_adaptation()

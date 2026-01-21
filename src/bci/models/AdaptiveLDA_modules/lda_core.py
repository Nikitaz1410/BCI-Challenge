"""
Core LDA classifier implementing standard discriminant analysis.

Implements LDA equations from Wu et al. (2024):
- D(x) = w^T * x + b  (decision function)
- w = Σ^(-1) * (μ2 - μ1)  (normal vector)
- b = -w^T * 0.5 * (μ1 + μ2)  (bias)
"""

import numpy as np


class LDACore:
    """
    Linear Discriminant Analysis classifier core.

    This implements the standard LDA equations for binary and multi-class
    classification without any adaptive updates.
    """

    def __init__(self):
        self.w = None  # Normal vector of hyperplane
        self.b = None  # Bias term
        self.mu = {}   # Class means {class_label: mean_vector}
        self.sigma = None  # Common covariance matrix
        self.sigma_inv = None  # Inverse covariance matrix
        self.classes = None  # Class labels

    def fit(self, X, y):
        """
        Fit LDA on features X and labels y.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        n_samples = len(X)

        # Compute class means (Equation 3)
        self.mu = {}
        for class_label in self.classes:
            self.mu[class_label] = np.mean(X[y == class_label], axis=0)

        # Compute pooled covariance matrix
        self.sigma = np.zeros((n_features, n_features))
        for class_label in self.classes:
            X_class = X[y == class_label]
            X_centered = X_class - self.mu[class_label]
            self.sigma += X_centered.T @ X_centered

        self.sigma /= (n_samples - n_classes)

        # Compute inverse
        self.sigma_inv = np.linalg.inv(self.sigma)

        # Compute LDA parameters (Equations 2 and 3)
        self._update_parameters()

    def _update_parameters(self):
        """Update w and b from current mu and sigma_inv."""
        if len(self.classes) == 2:
            mu1 = self.mu[self.classes[0]]
            mu2 = self.mu[self.classes[1]]
            self.w = self.sigma_inv @ (mu2 - mu1)
            self.b = -self.w.T @ (0.5 * (mu1 + mu2))

    def predict(self, X):
        """
        Predict class labels.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
        """
        if X.ndim == 1:
            X = X[np.newaxis, ...]
            single = True
        else:
            single = False

        n = X.shape[0]
        preds = np.zeros(n, dtype=int)

        if len(self.classes) == 2:
            for i in range(n):
                decision = self.w.T @ X[i] + self.b
                preds[i] = self.classes[1] if decision > 0 else self.classes[0]
        else:
            for i in range(n):
                dists = {c: (X[i] - self.mu[c]).T @ self.sigma_inv @ (X[i] - self.mu[c])
                        for c in self.classes}
                preds[i] = min(dists, key=dists.get)

        return preds[0] if single else preds

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns:
        --------
        probabilities : np.ndarray, shape (n_samples, n_classes)
        """
        if X.ndim == 1:
            X = X[np.newaxis, ...]
            single = True
        else:
            single = False

        n = X.shape[0]
        probs = np.zeros((n, len(self.classes)))

        if len(self.classes) == 2:
            for i in range(n):
                decision = self.w.T @ X[i] + self.b
                p1 = 1 / (1 + np.exp(-decision))
                probs[i] = [1 - p1, p1]
        else:
            for i in range(n):
                dists = np.array([
                    (X[i] - self.mu[c]).T @ self.sigma_inv @ (X[i] - self.mu[c])
                    for c in self.classes
                ])
                exp_neg = np.exp(-dists)
                probs[i] = exp_neg / exp_neg.sum()

        return probs[0:1] if single else probs

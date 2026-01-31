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


# ============================================================================
# Functional API wrappers for compatibility with hybrid_lda.py
# ============================================================================

def fit_lda(X, y, reg=1e-2, shrinkage_alpha=None, compute_priors=False):
    """
    Fit LDA and return parameters in functional form.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray, shape (n_samples,)
        Class labels
    reg : float, default=1e-2
        Regularization coefficient for covariance matrix
    shrinkage_alpha : float or None, default=None
        Shrinkage parameter in [0, 1]. If None, uses reg instead.
    compute_priors : bool, default=False
        Whether to compute and return class priors
        
    Returns:
    --------
    means : dict
        Dictionary mapping class labels to mean vectors
    cov : np.ndarray
        Regularized covariance matrix
    inv_cov : np.ndarray
        Inverse of regularized covariance matrix
    classes : np.ndarray
        Array of unique class labels
    priors : np.ndarray or None
        Array of class priors if compute_priors=True, else None
    """
    # Create and fit LDA model
    lda = LDACore()
    lda.fit(X, y)
    
    # Apply regularization/shrinkage to covariance
    if shrinkage_alpha is not None:
        # Shrinkage: (1 - alpha) * cov + alpha * trace(cov)/n_features * I
        trace_cov = np.trace(lda.sigma)
        n_features = lda.sigma.shape[0]
        identity = np.eye(n_features)
        regularized_cov = (1 - shrinkage_alpha) * lda.sigma + shrinkage_alpha * (trace_cov / n_features) * identity
    else:
        # Regularization: add reg * I
        regularized_cov = lda.sigma + reg * np.eye(lda.sigma.shape[0])
    
    # Compute inverse
    try:
        inv_cov = np.linalg.inv(regularized_cov)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        inv_cov = np.linalg.pinv(regularized_cov)
    
    # Compute priors if requested
    priors = None
    if compute_priors:
        unique_labels, counts = np.unique(y, return_counts=True)
        priors = counts / len(y)
        # Ensure priors are in the same order as classes
        priors_dict = dict(zip(unique_labels, priors))
        priors = np.array([priors_dict[c] for c in lda.classes])
    
    return lda.mu, regularized_cov, inv_cov, lda.classes, priors


def predict_proba_lda(X, means, inv_cov, classes, priors=None, method='discriminant'):
    """
    Predict class probabilities using LDA parameters.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features) or (n_features,)
        Feature matrix
    means : dict
        Dictionary mapping class labels to mean vectors
    inv_cov : np.ndarray, shape (n_features, n_features)
        Inverse covariance matrix
    classes : np.ndarray
        Array of class labels
    priors : np.ndarray or None, default=None
        Class priors. If None, assumes uniform priors.
    method : str, default='discriminant'
        Method for computing probabilities ('discriminant' or 'mahalanobis')
        
    Returns:
    --------
    probabilities : np.ndarray, shape (n_samples, n_classes)
        Class probabilities
    """
    if X.ndim == 1:
        X = X[np.newaxis, :]
        single = True
    else:
        single = False
    
    n_samples = X.shape[0]
    n_classes = len(classes)
    probs = np.zeros((n_samples, n_classes))
    
    if method == 'discriminant':
        # Use discriminant scores (Mahalanobis distance with log-priors)
        for i in range(n_samples):
            scores = np.zeros(n_classes)
            for j, c in enumerate(classes):
                # Mahalanobis distance: (x - mu)^T * inv_cov * (x - mu)
                diff = X[i] - means[c]
                mahal_dist = diff.T @ inv_cov @ diff
                scores[j] = -0.5 * mahal_dist
                
                # Add log prior if provided
                if priors is not None:
                    scores[j] += np.log(priors[j] + 1e-10)
            
            # Softmax to convert scores to probabilities
            exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
            probs[i] = exp_scores / exp_scores.sum()
    else:
        # Fallback: use Mahalanobis distance directly
        for i in range(n_samples):
            dists = np.array([
                (X[i] - means[c]).T @ inv_cov @ (X[i] - means[c])
                for c in classes
            ])
            exp_neg = np.exp(-dists)
            probs[i] = exp_neg / exp_neg.sum()
    
    return probs[0:1] if single else probs


def predict_lda(X, means, inv_cov, classes, priors=None):
    """
    Predict class labels using LDA parameters.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features) or (n_features,)
        Feature matrix
    means : dict
        Dictionary mapping class labels to mean vectors
    inv_cov : np.ndarray, shape (n_features, n_features)
        Inverse covariance matrix
    classes : np.ndarray
        Array of class labels
    priors : np.ndarray or None, default=None
        Class priors
        
    Returns:
    --------
    predictions : np.ndarray, shape (n_samples,)
        Predicted class labels
    """
    probs = predict_proba_lda(X, means, inv_cov, classes, priors, method='discriminant')
    class_indices = np.argmax(probs, axis=1)
    return np.array([classes[i] for i in class_indices])


def compute_discriminant_scores(X, means, inv_cov, classes, priors=None):
    """
    Compute discriminant scores for each class.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features) or (n_features,)
        Feature matrix
    means : dict
        Dictionary mapping class labels to mean vectors
    inv_cov : np.ndarray, shape (n_features, n_features)
        Inverse covariance matrix
    classes : np.ndarray
        Array of class labels
    priors : np.ndarray or None, default=None
        Class priors
        
    Returns:
    --------
    scores : np.ndarray, shape (n_samples, n_classes)
        Discriminant scores for each class
    """
    if X.ndim == 1:
        X = X[np.newaxis, :]
        single = True
    else:
        single = False
    
    n_samples = X.shape[0]
    n_classes = len(classes)
    scores = np.zeros((n_samples, n_classes))
    
    for i in range(n_samples):
        for j, c in enumerate(classes):
            # Mahalanobis distance: (x - mu)^T * inv_cov * (x - mu)
            diff = X[i] - means[c]
            mahal_dist = diff.T @ inv_cov @ diff
            scores[i, j] = -0.5 * mahal_dist
            
            # Add log prior if provided
            if priors is not None:
                scores[i, j] += np.log(priors[j] + 1e-10)
    
    return scores[0:1] if single else scores


def softmax(x, axis=-1):
    """
    Compute softmax function.
    
    Parameters:
    -----------
    x : np.ndarray
        Input array
    axis : int, default=-1
        Axis along which to compute softmax
        
    Returns:
    --------
    softmax : np.ndarray
        Softmax probabilities
    """
    # Numerical stability: subtract max
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

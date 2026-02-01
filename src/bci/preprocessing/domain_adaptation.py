'''
Module provides domain adaptation methods. 
Includes implementations of:
- CORAL (Correlation Alignment)
- TCA (Transfer Component Analysis)
- SA (Subspace Alignment)
- Riemannian Alignment
'''

import numpy as np
from scipy.linalg import inv, sqrtm, eigh
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, IncrementalPCA
from pyriemann.utils.mean import mean_riemann

from bci.models.riemann import compute_covariances, recentering


class CORAL(BaseEstimator, TransformerMixin):
    """
    CORAL: Correlation Alignment for Domain Adaptation.
    Works on feature vectors (e.g., Log-PSD, CSP features). 
    
    Reference
    ---------
    Sun et al. (2016). "Correlation Alignment for Unsupervised Domain Adaptation." Available: https://arxiv.org/abs/1612.01939
    """
    def __init__(self, lambda_reg=1.0):
        self.lambda_reg = lambda_reg
        self.target_cov_sqrt = None
        self.source_cov_inv_sqrt = None

    def fit(self, X_source, X_target):
        """
        Learns the transformation to map Source -> Target.
        X_target is used unlabeled.
        """
        # Add regularization to ensure matrices are invertible
        eye = np.eye(X_source.shape[1]) * self.lambda_reg
        
        cov_s = np.cov(X_source, rowvar=False) + eye
        cov_t = np.cov(X_target, rowvar=False) + eye

        # Whiten Source, then re-color with Target
        self.source_cov_inv_sqrt = inv(sqrtm(cov_s))
        self.target_cov_sqrt = sqrtm(cov_t)
        return self

    def transform(self, X):
        """Applies the alignment to the provided data (usually X_train)."""
        X_aligned = X @ self.source_cov_inv_sqrt @ self.target_cov_sqrt
        return np.real(X_aligned)
    

class TCA:
    """
    Transfer Component Analysis (TCA) for Domain Adaptation.
    
    Aligns source and target distributions by minimizing Maximum Mean 
    Discrepancy (MMD) in a reproducing kernel Hilbert space.

    Works on feature vectors (e.g., Log-PSD, CSP features). 
    
    Reference
    ---------
    Pan et al. (2011). "Domain Adaptation via Transfer Component Analysis."
    IEEE Transactions on Neural Networks, 22(2), 199-210. Available: https://pubmed.ncbi.nlm.nih.gov/21095864/
    """

    def __init__(self, n_components=30, mu=1.0, kernel_type='linear', gamma: float | None = 1.0):
        """
        Parameters
        ----------
        n_components : int, default=30
            Number of components in the latent subspace
        mu : float, default=1.0
            Regularization parameter (trade-off)
        kernel_type : str, default='linear'
            Kernel type: 'linear' or 'rbf'
        gamma : float, default=1.0|None
            Kernel parameter for RBF. Should be None for linear kernel.
        """
        self.n_components = n_components
        self.mu = mu
        self.kernel_type = kernel_type
        self.gamma = gamma if kernel_type == 'rbf' else None
        self.A = None
        self.X_fit_ = None
        self.X_mean = None
        self.ns = None
        self.nt = None

    def fit(self, Xs, Xt):
        """
        Learn TCA transformation.
        
        Parameters
        ----------
        Xs : ndarray, shape (n_source, n_features)
            Source domain data
        Xt : ndarray, shape (n_target, n_features)
            Target domain data (unlabeled)
        
        Returns
        -------
        self
        """
        self.ns = len(Xs)
        self.nt = len(Xt)
        n = self.ns + self.nt

        # Concatenate and center data
        X = np.vstack((Xs, Xt))
        self.X_mean = X.mean(axis=0)
        self.X_fit_ = X - self.X_mean

        # Construct MMD matrix (L)
        e = np.vstack((
            1.0 / self.ns * np.ones((self.ns, 1)),
            -1.0 / self.nt * np.ones((self.nt, 1))
        ))
        L = e @ e.T
        L = L / np.linalg.norm(L, 'fro')

        # Construct centering matrix (H)
        H = np.eye(n) - 1.0 / n * np.ones((n, n))

        # Compute kernel matrix
        kwargs = {}
        if self.kernel_type == 'rbf':
            kwargs['gamma'] = self.gamma
        
        K = pairwise_kernels(
            self.X_fit_, 
            metric=self.kernel_type, 
            filter_params=True, 
            **kwargs
        )

        # Solve generalized eigenvalue problem
        # (K L K + μI) W = λ (K H K) W
        A = K @ L @ K + self.mu * np.eye(n)
        B = K @ H @ K + 1e-6 * np.eye(n)  # Numerical stability

        # Use eigh (already imported at top)
        eigvals, eigvecs = eigh(A, B)

        # Sort eigenvalues (ascending = better alignment)
        idx = np.argsort(eigvals)
        self.A = eigvecs[:, idx[:self.n_components]]

        return self

    def transform(self, X):
        """
        Transform data to TCA subspace.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to transform
        
        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_components)
            Transformed data
        """
        if self.A is None:
            raise ValueError("Must fit() before transform()")

        # Center using training mean
        X_centered = X - self.X_mean

        # Compute kernel with training data
        kwargs = {}
        if self.kernel_type == 'rbf':
            kwargs['gamma'] = self.gamma

        K_new = pairwise_kernels(
            X_centered, 
            self.X_fit_,
            metric=self.kernel_type,
            **kwargs
        )

        # Project to subspace
        Z = K_new @ self.A
        return Z

    
 
class SA:
    """
    Subspace Alignment (SA) for Domain Adaptation.
    Aligns source and target subspaces via a linear transformation.

    Works on feature vectors (e.g., Log-PSD, CSP features).

    Reference
    ---------
    Fernando et al. (2014). "Subspace Alignment for Domain Adaptation". Available: https://arxiv.org/abs/1409.5241
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca_source = PCA(n_components=n_components)
        self.pca_target = PCA(n_components=n_components)
        self.Xs = None
        self.Xt = None
        self.M = None

    def fit(self, X_S, X_T):
        # 1. Fit Source PCA
        self.pca_source.fit(X_S)
        self.Xs = self.pca_source.components_.T  # Shape: (D, d)
        
        # 2. Fit Target PCA
        self.pca_target.fit(X_T)
        self.Xt = self.pca_target.components_.T  # Shape: (D, d)
        
        # 3. Compute Alignment Matrix
        # M = Xs' * Xt  -> Shape: (d, d)
        self.M = np.dot(self.Xs.T, self.Xt)
        
        return self

    def transform(self, X, domain="source"):
        if domain == "source":
            # Fix: Must project into subspace Xs first, THEN align with M
            # Step 1: Project X (N, D) -> Subspace (N, d)
            X_projected = np.dot(X, self.Xs) 
            
            # Step 2: Align Subspace (N, d) * M (d, d) -> (N, d)
            return np.dot(X_projected, self.M)
            
        elif domain == "target":
            # Target just needs projection into its own subspace
            # X (N, D) * Xt (D, d) -> (N, d)
            return np.dot(X, self.Xt)


class IncrementalSA:    # TODO: For large datasets. Tbd need
    def __init__(self, n_components):
        self.n_components = n_components
        # Using IncrementalPCA instead of standard PCA
        self.pca_source = IncrementalPCA(n_components=n_components)
        self.pca_target = IncrementalPCA(n_components=n_components)
        self.M = None
        self.Xs = None
        self.Xt = None

    def fit_source_batch(self, X_batch):
        """Call this on chunks of source data"""
        self.pca_source.partial_fit(X_batch)

    def fit_target_batch(self, X_batch):
        """Call this on chunks of target data"""
        self.pca_target.partial_fit(X_batch)

    def finalize_alignment(self):
        """Call this once all batches have been processed"""
        # Extract the learned bases
        self.Xs = self.pca_source.components_.T  # (Features, d)
        self.Xt = self.pca_target.components_.T  # (Features, d)
        
        # Compute M = Xs' * Xt
        self.M = np.dot(self.Xs.T, self.Xt)
        return self

    def transform(self, X, domain="source"):
        if self.M is None:
            raise ValueError("Must call finalize_alignment() before transform")
            
        if domain == "source":
            # Center using the running mean learned by IncrementalPCA
            X_projected = np.dot(X, self.Xs) 
            
            # Step 2: Align Subspace (N, d) * M (d, d) -> (N, d)
            return np.dot(X_projected, self.M)
        else:
            return np.dot(X, self.Xt)
        


def RiemannianAlignment(X_train, X_test, X_ref, cov_est=""):
    """
    Align training and test data to a reference (e.g. calibration data) using Riemannian recentering.

    Works on covariance matrices computed from time-series data (e.g., EEG).
    
    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_channels, n_times)
        Training time-series data
    X_test : ndarray, shape (n_test, n_channels, n_times)
        Test time-series data
    X_ref : ndarray, shape (n_ref, n_channels, n_times)
        Reference/calibration time-series data. Can also be training data (e.g. for cross-validation).
    cov_est : str
        Covariance estimator ('lwf' or '').
        If empty string, defaults to direct covariance computation.
    
    Returns
    -------
    C_train_centered : ndarray, shape (n_train, n_channels, n_channels)
        Aligned training covariances
    C_test_centered : ndarray, shape (n_test, n_channels, n_channels)
        Aligned test covariances
    """

    # Compute covariances
    C_train = compute_covariances(X_train, cov_est=cov_est)
    C_test = compute_covariances(X_test, cov_est=cov_est)
    C_calib = compute_covariances(X_ref, cov_est=cov_est)

    C_ref = mean_riemann(C_calib)
    # C_ref += 1e-6 * np.eye(C_ref.shape[0])

    # Recenter
    C_train_centered = recentering(C_train, C_ref)
    C_test_centered = recentering(C_test, C_ref)

    return C_train_centered, C_test_centered    




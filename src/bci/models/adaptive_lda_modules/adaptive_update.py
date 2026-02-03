"""
Adaptive update mechanism for LDA classifier.

Implements online adaptation using update coefficients (UC) following
Equations 4-11 from Wu et al. (2024).
"""

import numpy as np


def update_mean(mu_old, x_new, uc_mu):
    """
    Update class mean with new sample (Equation 4).

    μ_i(t) = (1 - UC_μ) * μ_i(t-1) + UC_μ * x_i(t)

    Parameters:
    -----------
    mu_old : np.ndarray
        Previous mean vector
    x_new : np.ndarray
        New feature vector
    uc_mu : float
        Update coefficient for mean

    Returns:
    --------
    mu_new : np.ndarray
        Updated mean vector
    """
    return (1 - uc_mu) * mu_old + uc_mu * x_new


def update_covariance(sigma_old, x_new, mu_old, uc_sigma, n_classes):
    """
    Update covariance matrix with new sample (Equation 5).

    Σ(t) = (1 - UC_Σ) * Σ(t-1) + (1/(N-1)) * UC_Σ * (x - μ_old) * (x - μ_old)^T

    Parameters:
    -----------
    sigma_old : np.ndarray
        Previous covariance matrix
    x_new : np.ndarray
        New feature vector
    mu_old : np.ndarray
        Previous mean vector (before update)
    uc_sigma : float
        Update coefficient for covariance
    n_classes : int
        Number of classes

    Returns:
    --------
    sigma_new : np.ndarray
        Updated covariance matrix
    """
    diff = x_new - mu_old
    outer_prod = np.outer(diff, diff)
    return (1 - uc_sigma) * sigma_old + (uc_sigma / (n_classes - 1)) * outer_prod


def update_covariance_inv_woodbury(sigma_inv_old, x_new, mu_old, uc_sigma, n_classes):
    """
    Update inverse covariance using Woodbury matrix identity (Equations 6-11).

    This avoids explicit matrix inversion at each update, making it efficient
    for online adaptation.

    Woodbury identity:
    (A + UU^T)^(-1) = A^(-1) - A^(-1)U(I + U^T A^(-1)U)^(-1)U^T A^(-1)

    Parameters:
    -----------
    sigma_inv_old : np.ndarray
        Previous inverse covariance matrix
    x_new : np.ndarray
        New feature vector
    mu_old : np.ndarray
        Previous mean vector
    uc_sigma : float
        Update coefficient for covariance
    n_classes : int
        Number of classes

    Returns:
    --------
    sigma_inv_new : np.ndarray
        Updated inverse covariance matrix
    """
    # Compute difference vector
    diff = x_new - mu_old

    # Scale for the inverse (Equation 9)
    A_inv = sigma_inv_old / (1 - uc_sigma)

    # Compute U and V for Woodbury identity (Equations 10-11)
    scale_factor = np.sqrt(uc_sigma / (n_classes - 1))
    U = scale_factor * diff
    V = U.copy()

    # Apply Woodbury identity (Equation 7)
    inner_term = 1 + V.T @ A_inv @ U

    # Compute the update
    outer = np.outer(U, V)
    sigma_inv_new = A_inv - (A_inv @ outer @ A_inv) / inner_term

    return sigma_inv_new

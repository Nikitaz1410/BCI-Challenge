import numpy as np
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.tangentspace import log_map_riemann, tangent_space
from pyriemann.utils.test import is_sym_pos_def


class RiemannianClf:
    def __init__(self):
        self.clf = FgMDM()

    # Fit model
    def fit(self, covs, y):
        self.clf.fit(covs, y)

    # Predict labels
    def predict(self, cov):
        prediction = self.clf.predict(cov)
        return prediction

    # Predict Probabilities
    def predict_proba(self, cov):
        proba = self.clf.predict_proba(cov)
        return proba

    # Update centroids -> TODO: For online alignment
    def _update_centroids(self, covs, y):
        self.clf.fit(covs, y)


# Center the data around the given reference
def recentering(covs, reference):
    inv_sqrt_ref = invsqrtm(reference)
    if len(covs.shape) == 2:
        recentered_covs = inv_sqrt_ref @ covs @ inv_sqrt_ref
        if not is_sym_pos_def(recentered_covs):
            raise ValueError(
                "Recentered covariance is not symmetric positive definite."
            )
    else:
        recentered_covs = np.array([inv_sqrt_ref @ cov @ inv_sqrt_ref for cov in covs])
        if not all(is_sym_pos_def(cov) for cov in recentered_covs):
            raise ValueError(
                "At least one recentered covariance is not symmetric positive definite."
            )

    return recentered_covs


# Move the data to the tangent space of the given reference
def project_to_tangent_space(covs, reference, mode="vector"):
    # Default metric is the affine-invariant riemann metric (AIRM)
    if mode == "matrix":
        ts_covs = log_map_riemann(reference, covs)
    elif mode == "vector":
        ts_covs = tangent_space(covs, reference)
    else:
        raise ValueError("Mode should be either 'matrix' or 'vector'.")

    return ts_covs


# Extract covariance matrices
def compute_covariances(data: np.ndarray, cov_est: str = "") -> np.ndarray:
    """
    Compute the covariance matrix of the given data.
    """
    _, n_channels, n_samples = data.shape[0], data.shape[1], data.shape[2]

    covs = np.full((n_samples, n_channels, n_channels), 0.0)
    if len(cov_est) == 0:
        print("Using direct covariance computation.")
        for i, window in enumerate(data):
            # Direct computation of the covariance matrix
            data -= np.mean(window, axis=0)
            t_cov = data.T @ data
            cov = t_cov / np.trace(t_cov)
            covs[i] = cov
    else:
        print(f"Using covariance estimator: {cov_est}")
        cov_est = Covariances(estimator=cov_est)
        covs = cov_est.fit_transform(data)
        covs = covs

    if not all(is_sym_pos_def(cov) for cov in covs):
        raise ValueError(
            "At least one recentered covariance is not symmetric positive definite."
        )

    return covs

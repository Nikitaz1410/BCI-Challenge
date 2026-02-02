import pickle

import numpy as np
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.tangentspace import log_map_riemann, tangent_space
from pyriemann.utils.test import is_sym_pos_def
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.geodesic import geodesic_riemann
from sklearn.externals._packaging.version import Optional
from sklearn.externals.array_api_compat.numpy import bool_, sign


# RiemannianClf: Classifier using Riemannian geometry for covariance matrices
class RiemannianClf:
    def __init__(self, cov_est=""):
        """
        Initialize the classifier.

        Parameters:
        cov_est (str): Covariance estimator to use. If empty, uses direct computation.
        """
        self.clf = FgMDM()  # Initialize the FgMDM classifier from pyriemann
        self.cov_est = cov_est  # Store the covariance estimator type
        self.online_predictions = 0  # Needed for realtime adaptation

    def fit(self, signals, y):
        """
        Fit the classifier to the provided signals and labels.

        Parameters:
        signals (np.ndarray): Input signals (trials x channels x samples or channels x samples)
        y (np.ndarray): Labels for each trial
        """
        covs = self._extract_features(signals)
        # TODO: Add recentering? (Normalization)
        self.clf.fit(covs, y)

    # Approach
    # Create class centroids for each subject/session
    # Optional: Recenter Data to Baseline - Session Centroid (Inter-Subject/Session Variability)
    # Train the model on the class centroids
    # Optional: Finetune to one of Fina's sessions (Recentering Formula, Geodesic Interpolation)
    def fit_centered(self, signals, y, groups):
        covs = self._extract_features(signals)

        unique_groups = np.unique(groups)
        classes = np.unique(y)

        centroids = []
        centroids_y = []

        for group_id, group in enumerate(unique_groups):
            # Compute the centroids of the group (sessions | subjects)
            signal_idx = np.where(groups == group)[0]

            grouped_covs = covs[signal_idx, :, :]
            grouped_y = y[signal_idx]

            # Add recentering to session centroid to have better class separation
            centroid = mean_riemann(grouped_covs)

            # Compute the centered covs
            recentered_covs = recentering(grouped_covs, centroid)

            for cls_id, cls in enumerate(classes):
                try:
                    cls_covs = recentered_covs[grouped_y == cls]
                    cls_covs = np.array(cls_covs)

                    if cls_covs.shape[0] <= 0:
                        print(f"No covs for {cls} of sess {group}")
                    else:
                        cls_centroid = mean_riemann(cls_covs)

                        if not is_sym_pos_def(cls_centroid):
                            print(f"Centroid of {cls}, {group} is not SPD!")
                        else:
                            # Compute the class-centroid
                            centroids.append(cls_centroid)
                            centroids_y.append(cls)
                except Exception as e:
                    print(f"Problem with Centroid of {cls}, {group}: {e}")

        # Fit the classifier based on the
        centroids_X = np.array(centroids)
        centroids_y = np.array(centroids_y)

        print(
            f"Training on {centroids_X.shape} centroids with {centroids_y.shape} labels."
        )
        self.clf.fit(centroids_X, centroids_y)

    def predict(self, cov):
        """
        Predict class labels for the provided covariance matrices.

        Parameters:
        cov (np.ndarray): Input signals or covariance matrices

        Returns:
        np.ndarray: Predicted class labels
        """
        cov = self._extract_features(cov)
        return self.clf.predict(cov)

    def predict_proba(self, cov):
        """
        Predict class probabilities for the provided covariance matrices.

        Parameters:
        cov (np.ndarray): Input signals or covariance matrices

        Returns:
        np.ndarray: Predicted class probabilities
        """
        try:
            # `_extract_features` returns covariance matrices of shape:
            # - (n_trials, n_channels, n_channels) for batch input
            # - (n_channels, n_channels) for a single trial
            cov = self._extract_features(cov)
            if cov.ndim == 2:
                cov = cov[np.newaxis, ...]
            return self.clf.predict_proba(cov), cov
        except ValueError as e:
            print(f"Error in feature extraction: {e}")
            return None, None

    # Adaptation based on: Towards Adaptive Classification using Riemannian Geometry approaches in Brain-Computer Interfaces
    def adapt(self, cov, prediction):
        """
        Adapt the classifier by moving the class centroid towards the input covariance matrix.

        This method implements online adaptation by adjusting the class centroid in Riemannian space
        based on the current prediction. The adaptation factor decreases with each update to
        gradually incorporate new information while maintaining stability.

        Parameters:
        cov (np.ndarray): Input covariance matrix to adapt towards
        prediction (int): Predicted class label for the input covariance
        """

        # Move the class centroid in the riemann space towards the handled covariance
        # The class is the predicted class (How do we explain this?)

        self.online_predictions += 1

        adaptation_factor = 1 / (
            self.online_predictions + 1
        )  # Number of already used predictions for adaptation

        # print(f"Move Towards: {prediction} by {adaptation_factor} geodesic units.")

        cls_centroid = self.clf._mdm.covmeans_[prediction]
        moved_cls_centroid = geodesic_riemann(cls_centroid, cov, adaptation_factor)

        if is_sym_pos_def(moved_cls_centroid):
            # Replace the centroid of the clf
            self.clf._mdm.covmeans_[prediction] = moved_cls_centroid
        else:
            print("Moved centroid is not SPD!")

    def _extract_features(self, signals):
        """
        Extract covariance features from the input signals.

        Parameters:
        signals (np.ndarray): Input signals

        Returns:
        np.ndarray: Covariance matrices
        """
        return compute_covariances(signals, self.cov_est)

    def save(self, filepath):
        """
        Save the classifier instance to a file using pickle.

        Parameters:
        filepath (str): Path to the file where the instance will be saved
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """
        Load a classifier instance from a file.

        Parameters:
        filepath (str): Path to the file from which to load the instance

        Returns:
        RiemannianClf: Loaded classifier instance
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)


def recentering(covs, reference):
    """
    Center the covariance matrices around the given reference.

    Parameters:
    covs (np.ndarray): Covariance matrix or array of covariance matrices
    reference (np.ndarray): Reference covariance matrix

    Returns:
    np.ndarray: Recentered covariance matrix/matrices

    Raises:
    ValueError: If any recentered covariance is not symmetric positive definite
    """
    inv_sqrt_ref = invsqrtm(reference)
    if len(covs.shape) == 2:
        # Single covariance matrix
        recentered_covs = inv_sqrt_ref @ covs @ inv_sqrt_ref
        if not is_sym_pos_def(recentered_covs):
            raise ValueError(
                "Recentered covariance is not symmetric positive definite."
            )
    else:
        # Multiple covariance matrices
        recentered_covs = np.array([inv_sqrt_ref @ cov @ inv_sqrt_ref for cov in covs])
        if not all(is_sym_pos_def(cov) for cov in recentered_covs):
            raise ValueError(
                "At least one recentered covariance is not symmetric positive definite."
            )

    return recentered_covs


def project_to_tangent_space(covs, reference, mode="vector"):
    """
    Project covariance matrices to the tangent space of the given reference.

    Parameters:
    covs (np.ndarray): Covariance matrix or array of covariance matrices
    reference (np.ndarray): Reference covariance matrix
    mode (str): 'matrix' for matrix logarithm, 'vector' for tangent space vectorization

    Returns:
    np.ndarray: Projected data in tangent space

    Raises:
    ValueError: If mode is not 'matrix' or 'vector'
    """
    # Default metric is the affine-invariant riemann metric (AIRM)
    if mode == "matrix":
        ts_covs = log_map_riemann(reference, covs)
    elif mode == "vector":
        ts_covs = tangent_space(covs, reference)
    else:
        raise ValueError("Mode should be either 'matrix' or 'vector'.")

    return ts_covs


def compute_covariances(data: np.ndarray, cov_est: str = "") -> np.ndarray:
    """
    Compute the covariance matrix/matrices of the given data.

    Parameters:
    data (np.ndarray): 2D array (channels x samples) or 3D array (trials x channels x samples)
    cov_est (str): Covariance estimator to use. If empty, uses direct computation.

    Returns:
    np.ndarray: Single covariance matrix (2D) or array of covariance matrices (3D)

    Raises:
    ValueError: If any computed covariance is not symmetric positive definite
    """
    # If input is 2D, make it 3D with a single trial for unified processing
    is_single = False
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        is_single = True

    n_trials, n_channels, n_samples = (
        data.shape
    )  # n_samples is not used, but kept for clarity

    if len(cov_est) == 0:
        # print("Using direct covariance computation.")
        covs = np.empty((n_trials, n_channels, n_channels))
        for i, window in enumerate(data):
            # Remove mean from each channel to center the data
            window = window - np.mean(window, axis=1, keepdims=True)
            t_cov = window @ window.T  # Compute covariance
            cov = t_cov / np.trace(t_cov)  # Normalize by trace
            covs[i] = cov
    else:
        # print(f"Using covariance estimator: {cov_est}")
        cov_estimator = Covariances(estimator=cov_est)
        covs = cov_estimator.fit_transform(data)

    # Check that all covariance matrices are symmetric positive definite
    if not all(is_sym_pos_def(cov) for cov in covs):
        raise ValueError(
            "At least one recentered covariance is not symmetric positive definite."
        )

    # If input was 2D, return 2D output
    if is_single:
        return covs[0]
    return covs

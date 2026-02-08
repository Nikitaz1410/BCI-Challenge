import pickle

import numpy as np
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.test import is_sym_pos_def

# --- Core Riemannian Classifier Class ---


class RiemannianClf:
    """
    Classifier using Riemannian geometry for covariance matrices.

    This class wraps pyriemann's FgMDM and provides utilities for session-based
    recentering and online distribution adaptation.
    """

    def __init__(self, cov_est=""):
        """
        Initializes the Riemannian pipeline.

        Args:
            cov_est (str): Covariance estimator type (e.g., 'scm', 'lwf', 'oas').
                           If empty, uses direct computation with trace normalization.
        """
        self.clf = FgMDM()
        self.cov_est = cov_est
        self.online_predictions = 0  # Counter for adaptation decay
        self.recentering_centroid = None  # Global/Session mean for normalization

    def fit(self, signals, y):
        """Standard fit: Extracts covariances and trains the FgMDM model."""
        covs = self._extract_features(signals)
        self.clf.fit(covs, y)

    def fit_centered(self, signals, y, groups):
        """
        Fits the model after recentering data by group (e.g., by subject or session).

        Recentering moves each group's mean to the identity matrix, reducing
        inter-subject/inter-session variability.
        """
        covs = self._extract_features(signals)
        unique_groups = np.unique(groups)

        X_train, y_train = None, None

        for group in unique_groups:
            idx = np.where(groups == group)[0]
            grouped_covs = covs[idx]
            y_grouped = y[idx]

            # 1. Compute the Riemannian Mean for this group
            centroid = mean_riemann(grouped_covs)

            # 2. Project group data to center (Parallel Transport to Identity)
            recentered_covs = recentering(grouped_covs, centroid)

            if X_train is None:
                X_train, y_train = recentered_covs, y_grouped
            else:
                X_train = np.concatenate((X_train, recentered_covs), axis=0)
                y_train = np.concatenate((y_train, y_grouped), axis=0)

        # Set the global centroid for future prediction-time centering
        self.recentering_centroid = mean_riemann(X_train)
        self.clf.fit(X_train, y_train)

    def fit_special(self, signals, y, groups):
        """
        Advanced fit: Trains specifically on class centroids per group.

        Reduces the dataset to 'Representative Archetypes' (one mean per class per group)
        to handle high noise or imbalanced session data.
        """
        covs = self._extract_features(signals)
        unique_groups = np.unique(groups)
        classes = np.unique(y)

        centroids, centroids_y = [], []

        for group in unique_groups:
            idx = np.where(groups == group)[0]
            grouped_covs = covs[idx]
            grouped_y = y[idx]

            # Recenter session data to common manifold origin
            group_mean = mean_riemann(grouped_covs)
            recentered_covs = recentering(grouped_covs, group_mean)

            for cls in classes:
                cls_covs = recentered_covs[grouped_y == cls]
                if cls_covs.shape[0] > 0:
                    cls_centroid = mean_riemann(cls_covs)

                    if is_sym_pos_def(cls_centroid):
                        centroids.append(cls_centroid)
                        centroids_y.append(cls)
                    else:
                        print(
                            f"Warning: Centroid for Class {cls}, Group {group} is not SPD."
                        )

        centroids_X = np.array(centroids)
        centroids_y = np.array(centroids_y)

        self.recentering_centroid = mean_riemann(centroids_X)
        print(f"Training on {centroids_X.shape[0]} class-centroids.")
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
        return self.clf.predict(cov), cov

    def predict_proba(self, cov):
        """
        Predict class probabilities for the provided covariance matrices.

        Parameters:
        cov (np.ndarray): Input signals or covariance matrices

        Returns:
        np.ndarray: Predicted class probabilities
        """
        try:
            cov = self._extract_features(cov)

            if cov.ndim == 2:
                cov = cov[np.newaxis, ...]

            return self.clf.predict_proba(cov), cov

        except ValueError as e:
            print(f"Error in feature extraction: {e}")

            return None, None

    def predict_with_recentering(self, signals):
        """
        Predicts labels after applying online adaptation and recentering.

        Args:
            signals: Raw signals
        Returns:
            tuple: (Predictions, Probabilities, Recentered_Covariances)
        """
        try:
            covs = self._extract_features(signals)
            if covs.ndim == 2:
                covs = covs[np.newaxis, ...]

            preds, probs, recentered_list = [], [], []

            for cov in covs:
                # 1. Update global distribution mean (Unsupervised Adaptation)
                self.adapt_distribution(cov[np.newaxis, ...])

                # 2. Recenter the sample
                rc_cov = recentering(cov, self.recentering_centroid)

                # 3. Classify
                preds.append(self.clf.predict(rc_cov)[0])
                probs.append(self.clf.predict_proba(rc_cov)[0])
                recentered_list.append(rc_cov[0])

            return np.array(preds), np.array(probs), np.array(recentered_list)
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None, None, None

    def adapt_distribution(self, cov):
        """
        Unsupervised Adaptation: Slides the recentering centroid toward new data.

        Uses the Geodesic (the shortest path on the manifold) to update the
        global mean with a decaying learning rate.
        """
        self.online_predictions += 1
        alpha = 1 / (self.online_predictions + 1)

        new_rc = geodesic_riemann(self.recentering_centroid, cov, alpha)
        if is_sym_pos_def(new_rc):
            self.recentering_centroid = new_rc

    def adapt(self, cov, prediction):
        """
        Supervised Adaptation: Moves a specific class centroid toward a sample.

        Based on:
        Kumar, Satyam, Florian Yger, and Fabien Lotte.
        "Towards adaptive classification using Riemannian geometry approaches in brain-computer interfaces."
        2019 7th International Winter Conference on Brain-Computer Interface (BCI). IEEE, 2019.

        Args:
            cov: The sample covariance.
            prediction: The predicted (or ground truth) class label.
        """
        self.online_predictions += 1
        alpha = 1 / (self.online_predictions + 1)

        # Access internal MDM class centroids
        cls_centroid = self.clf._mdm.covmeans_[prediction]
        moved_centroid = geodesic_riemann(cls_centroid, cov, alpha)

        if is_sym_pos_def(moved_centroid):
            self.clf._mdm.covmeans_[prediction] = moved_centroid

    def _extract_features(self, signals):
        """Internal helper to ensure input is converted to covariance matrices."""
        return compute_covariances(signals, self.cov_est)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)


# --- Riemannian Utility Functions ---
def recentering(covs, reference):
    """
    Applies the transformation $C' = R^{-1/2} C R^{-1/2}$.
    This centers the distribution around the Identity matrix.
    """
    inv_sqrt_ref = invsqrtm(reference)

    if covs.ndim == 2:
        res = inv_sqrt_ref @ covs @ inv_sqrt_ref
        if not is_sym_pos_def(res):
            raise ValueError("Resulting matrix not SPD")
        return res

    res = np.array([inv_sqrt_ref @ c @ inv_sqrt_ref for c in covs])
    return res


def compute_covariances(data: np.ndarray, cov_est: str = "") -> np.ndarray:
    """
    Computes covariance matrices. If no estimator is provided, uses
    sample covariance normalized by the trace.
    """
    is_single = data.ndim == 2
    if is_single:
        data = data[np.newaxis, ...]

    if not cov_est:
        # Manual computation: Center -> Dot Product -> Trace Normalize
        n_trials, n_chan, _ = data.shape
        covs = np.empty((n_trials, n_chan, n_chan))
        for i, window in enumerate(data):
            window = window - np.mean(window, axis=1, keepdims=True)
            t_cov = window @ window.T
            covs[i] = t_cov / np.trace(t_cov)
    else:
        cov_estimator = Covariances(estimator=cov_est)
        covs = cov_estimator.fit_transform(data)

    if not all(is_sym_pos_def(c) for c in covs):
        raise ValueError("Non-SPD matrix detected in covariance computation.")

    return covs[0] if is_single else covs

import mne
import numpy as np
from autoreject import get_rejection_threshold
from typing import Optional, Any, Tuple
from scipy import stats


class ArtefactRemoval:
    def __init__(
        self,
        rejection_threshold: Optional[float] = None,
        threshold_multiplier: float = 1.0,
        max_bad_channels_pct: float = 0.5,
        use_amplitude: bool = True,
        use_variance: bool = True,
        use_gradient: bool = True,
        use_consistency: bool = True,
    ) -> None:
        """
        Initialize the ArtefactRemoval class with comprehensive artifact detection.

        Parameters:
        -----------
        rejection_threshold (Optional[float]): Manual threshold. If None, computed automatically.
        threshold_multiplier (float): Multiplier for computed thresholds. Higher = more lenient.
        max_bad_channels_pct (float): Max % of channels that can be bad before rejecting epoch.
        use_amplitude (bool): Check for amplitude outliers (clipping, large artifacts).
        use_variance (bool): Check for variance outliers (movement artifacts, muscle activity).
        use_gradient (bool): Check for sudden jumps (gradient-based, catches spikes).
        use_consistency (bool): Check channel consistency (isolated bad channels).
        """
        self.rejection_threshold = rejection_threshold
        self.threshold_multiplier = threshold_multiplier
        self.max_bad_channels_pct = max_bad_channels_pct

        # Feature flags
        self.use_amplitude = use_amplitude
        self.use_variance = use_variance
        self.use_gradient = use_gradient
        self.use_consistency = use_consistency

        # Computed thresholds
        self.amplitude_threshold = None
        self.variance_threshold = None
        self.gradient_threshold = None
        self.consistency_threshold = None

    def get_rejection_thresholds(
        self,
        epoch_data: np.ndarray,
        config: Any,
        use_percentile: bool = True,
        percentile: float = 95.0,
    ) -> None:
        """
        Compute rejection thresholds using multiple criteria.

        Parameters:
        -----------
        epoch_data: np.ndarray
            Shape (n_epochs, n_channels, n_times)
        config: Any
            Config object with 'channels' and 'fs' attributes
        use_percentile: bool
            If True, use percentile-based thresholds
        percentile: float
            Percentile to use (e.g., 95.0 = 95th percentile)
        """
        # Validate input
        if not hasattr(config, "channels") or not hasattr(config, "fs"):
            raise AttributeError("Config must have 'channels' and 'fs' attributes.")

        if not isinstance(epoch_data, np.ndarray):
            raise TypeError("epoch_data must be numpy ndarray.")

        n_epochs, n_channels, n_times = epoch_data.shape

        print(f"\n🔍 Computing Artifact Rejection Thresholds:")
        print(f"   Input: {n_epochs} epochs, {n_channels} channels, {n_times} time points")

        # 1. AMPLITUDE REJECTION (max absolute amplitude)
        if self.use_amplitude:
            # Max amplitude per epoch (max across all channels and time)
            max_amplitudes = np.max(np.abs(epoch_data), axis=(1, 2))  # (n_epochs,)
            self.amplitude_threshold = (
                np.percentile(max_amplitudes, percentile) * self.threshold_multiplier
            )
            print(
                f"   ✓ Amplitude threshold ({percentile}th percentile): "
                f"{np.percentile(max_amplitudes, percentile):.2f} μV"
            )
            print(
                f"      After multiplier ({self.threshold_multiplier}x): "
                f"{self.amplitude_threshold:.2f} μV"
            )

        # 2. VARIANCE REJECTION (high variance = movement/muscle artifacts)
        if self.use_variance:
            # Variance per epoch (variance across all channels and time)
            variances = np.var(epoch_data, axis=(1, 2))  # (n_epochs,)
            self.variance_threshold = (
                np.percentile(variances, percentile) * self.threshold_multiplier
            )
            print(
                f"   ✓ Variance threshold ({percentile}th percentile): "
                f"{np.percentile(variances, percentile):.4f}"
            )
            print(
                f"      After multiplier ({self.threshold_multiplier}x): "
                f"{self.variance_threshold:.4f}"
            )

        # 3. GRADIENT REJECTION (sudden jumps = spikes/artifacts)
        if self.use_gradient:
            # Compute gradient (difference between consecutive samples)
            # Gradient magnitude per epoch
            gradients = []
            for epoch in epoch_data:
                # Shape: (n_channels, n_times)
                grad = np.diff(epoch, axis=1)  # (n_channels, n_times-1)
                max_grad = np.max(np.abs(grad))  # Max gradient magnitude
                gradients.append(max_grad)
            gradients = np.array(gradients)
            self.gradient_threshold = (
                np.percentile(gradients, percentile) * self.threshold_multiplier
            )
            print(
                f"   ✓ Gradient threshold ({percentile}th percentile): "
                f"{np.percentile(gradients, percentile):.4f} μV/sample"
            )
            print(
                f"      After multiplier ({self.threshold_multiplier}x): "
                f"{self.gradient_threshold:.4f} μV/sample"
            )

        # 4. CONSISTENCY REJECTION (isolated bad channels)
        if self.use_consistency:
            # For each epoch, check if any channel deviates significantly from others
            # Use z-score: if a channel's variance is >3 std away from other channels
            consistency_scores = []
            for epoch in epoch_data:
                # Variance per channel: (n_channels,)
                channel_vars = np.var(epoch, axis=1)
                # Z-score of channel variances
                z_scores = np.abs(stats.zscore(channel_vars))
                max_z_score = np.max(z_scores)
                consistency_scores.append(max_z_score)
            consistency_scores = np.array(consistency_scores)
            # Threshold: 3 standard deviations (empirical rule)
            self.consistency_threshold = (
                np.percentile(consistency_scores, percentile) * self.threshold_multiplier
            )
            print(
                f"   ✓ Consistency threshold ({percentile}th percentile): "
                f"{np.percentile(consistency_scores, percentile):.2f} z-score"
            )
            print(
                f"      After multiplier ({self.threshold_multiplier}x): "
                f"{self.consistency_threshold:.2f} z-score"
            )

        # Set legacy rejection_threshold for backward compatibility
        if self.amplitude_threshold is not None:
            self.rejection_threshold = self.amplitude_threshold

    def reject_bad_epochs(
        self, epochs_data: np.ndarray, epochs_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reject bad epochs using multiple criteria.

        An epoch is rejected if it fails ANY of the enabled criteria.
        """
        if self.amplitude_threshold is None and not any(
            [
                self.use_amplitude,
                self.use_variance,
                self.use_gradient,
                self.use_consistency,
            ]
        ):
            raise ValueError("Rejection thresholds have not been computed yet.")

        if epochs_data.shape[0] != epochs_labels.shape[0]:
            raise ValueError("Number of epochs and labels must match.")

        n_epochs, n_channels, n_times = epochs_data.shape
        good_epochs = []
        good_labels = []
        rejection_reasons = {
            "amplitude": 0,
            "variance": 0,
            "gradient": 0,
            "consistency": 0,
            "too_many_bad_channels": 0,
        }

        for i in range(n_epochs):
            epoch = epochs_data[i]  # (n_channels, n_times)
            is_bad = False
            bad_channels = np.zeros(n_channels, dtype=bool)

            # Check each criterion
            if self.use_amplitude and self.amplitude_threshold is not None:
                channel_max_amps = np.max(np.abs(epoch), axis=1)  # (n_channels,)
                bad_amp = channel_max_amps > self.amplitude_threshold
                bad_channels = bad_channels | bad_amp
                if np.sum(bad_amp) > (n_channels * self.max_bad_channels_pct):
                    is_bad = True
                    rejection_reasons["amplitude"] += 1
                    continue

            if self.use_variance and self.variance_threshold is not None:
                channel_vars = np.var(epoch, axis=1)  # (n_channels,)
                bad_var = channel_vars > self.variance_threshold
                bad_channels = bad_channels | bad_var
                if np.sum(bad_var) > (n_channels * self.max_bad_channels_pct):
                    is_bad = True
                    rejection_reasons["variance"] += 1
                    continue

            if self.use_gradient and self.gradient_threshold is not None:
                grad = np.diff(epoch, axis=1)  # (n_channels, n_times-1)
                max_grads = np.max(np.abs(grad), axis=1)  # (n_channels,)
                bad_grad = max_grads > self.gradient_threshold
                bad_channels = bad_channels | bad_grad
                if np.sum(bad_grad) > (n_channels * self.max_bad_channels_pct):
                    is_bad = True
                    rejection_reasons["gradient"] += 1
                    continue

            if self.use_consistency and self.consistency_threshold is not None:
                channel_vars = np.var(epoch, axis=1)
                if len(channel_vars) > 2:  # Need at least 3 channels for z-score
                    z_scores = np.abs(stats.zscore(channel_vars))
                    if np.max(z_scores) > self.consistency_threshold:
                        is_bad = True
                        rejection_reasons["consistency"] += 1
                        continue

            # Final check: too many bad channels overall?
            if np.sum(bad_channels) > (n_channels * self.max_bad_channels_pct):
                is_bad = True
                rejection_reasons["too_many_bad_channels"] += 1
                continue

            # Epoch passed all checks
            if not is_bad:
                good_epochs.append(epoch)
                good_labels.append(epochs_labels[i])

        rejected = n_epochs - len(good_epochs)
        rejection_pct = (rejected / n_epochs * 100) if n_epochs > 0 else 0

        print(f"\n📊 Artifact Rejection Statistics:")
        print(f"   Total epochs: {n_epochs}")
        print(f"   Kept: {len(good_epochs)} ({100-rejection_pct:.1f}%)")
        print(f"   Rejected: {rejected} ({rejection_pct:.1f}%)")
        if rejected > 0:
            print(f"\n   Rejection reasons:")
            for reason, count in rejection_reasons.items():
                if count > 0:
                    print(f"     - {reason}: {count} epochs")

        return np.array(good_epochs), np.array(good_labels)

    def reject_bad_epoches_online(self, epoch: np.ndarray) -> None:
        """Placeholder for online epoch rejection."""
        pass

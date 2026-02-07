from typing import List, Optional, Union

import mne
import numpy as np
import scipy.signal

from bci.utils.bci_config import EEGConfig


class Filter:
    """
    A robust EEG signal filter supporting both batch (offline) and
    real-time (online) processing using SOS (Second-Order Sections).

    This class handles bandpass filtering and notch filtering,
    maintaining state buffers to prevent edge artifacts during
    streaming applications.
    """

    def __init__(
        self,
        config: EEGConfig,
        online: bool = False,
        n_channels_online: Optional[int] = None,
    ) -> None:
        """
        Initializes the filter parameters and state buffers.

        Args:
            config: Configuration object containing 'order', 'frequencies',
                    'fs', 'channels', and 'remove_channels'.
            online: If True, initializes 'zi' state for streaming data.
            n_channels_online: Explicit channel count for online state.
                               Defaults to (total - removed) channels.
        """
        self.config = config
        self.fs = config.fs

        # Define filter coefficients in Second-Order Sections (SOS)
        # for better numerical stability than (b, a)
        self.sos: np.ndarray = scipy.signal.butter(
            N=self.config.order,
            Wn=np.array(self.config.frequencies),
            btype="bandpass",
            fs=self.fs,
            output="sos",
        )

        self.zi: Optional[np.ndarray] = None
        if online:
            self._init_online_state(n_channels_online)

    def _init_online_state(self, n_channels: Optional[int]) -> None:
        """Initializes the filter state (zi) for each channel."""
        if n_channels is None:
            n_channels = len(self.config.channels) - len(self.config.remove_channels)

        # Get steady-state step response for SOS filter
        zi_base = scipy.signal.sosfilt_zi(self.sos)  # shape: (n_sections, 2)

        # Expand zi to handle all channels: (n_sections, n_channels, 2)
        self.zi = np.repeat(zi_base[:, np.newaxis, :], n_channels, axis=1)
        print(f"Initialized online filter state: {self.zi.shape}")

    def apply_filter_offline(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies bandpass filter to a full data array.

        Args:
            signal: Input array of shape (n_channels, n_time).

        Returns:
            Filtered signal array.
        """
        return scipy.signal.sosfilt(self.sos, signal, axis=-1)

    def apply_filter_online(self, data_chunk: np.ndarray) -> np.ndarray:
        """
        Applies filter to a streaming data chunk and updates internal state.

        Args:
            data_chunk: Small chunk of EEG data (n_channels, n_samples).

        Returns:
            Filtered chunk of the same shape.
        """
        if self.zi is None:
            raise ValueError("Filter was not initialized for online mode.")

        # sosfilt updates zi in-place, ensuring continuity between chunks
        filtered_chunk, self.zi = scipy.signal.sosfilt(
            self.sos, data_chunk, zi=self.zi, axis=-1
        )
        return filtered_chunk

    def apply_notch(
        self, raw: mne.io.BaseRaw, freqs: Union[float, List[float]]
    ) -> mne.io.BaseRaw:
        """
        Applies a notch filter using MNE's built-in functionality.

        Args:
            raw: MNE Raw object.
            freqs: Frequency or list of frequencies (e.g., 50.0 for power line noise).

        Returns:
            The MNE Raw object with applied notch filters.
        """
        raw.notch_filter(freqs=freqs, method="iir", phase="forward")
        return raw

    def get_filter_latency(self) -> float:
        """
        Calculates the average group delay (latency) within the passband.

        Returns:
            Average latency in milliseconds.
        """
        # Convert SOS to Transfer Function for delay calculation
        b, a = scipy.signal.sos2tf(self.sos)
        w, gd = scipy.signal.group_delay((b, a), fs=self.fs)

        # Focus calculation on the passband frequencies
        low, high = self.config.frequencies
        passband_mask = (w >= low) & (w <= high)

        if not np.any(passband_mask):
            avg_delay_samples = np.mean(gd)
        else:
            avg_delay_samples = np.mean(gd[passband_mask])

        avg_latency_ms = (avg_delay_samples / self.fs) * 1000
        return avg_latency_ms

    def reset_state(self) -> None:
        """Resets the online filter state to zeros/steady-state."""
        if self.zi is not None:
            self._init_online_state(self.zi.shape[1])

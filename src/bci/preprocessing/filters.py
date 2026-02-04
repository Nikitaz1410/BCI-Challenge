# change: New module extracted from preprocessing.py to centralize filter helpers
"""
Filter helpers: parameter parsing, factory, latency estimation, and dataset filtering helper.
This keeps filtering logic in a single file for clarity and reuse.
"""

from typing import Optional, Tuple

import mne
import numpy as np
import scipy
import scipy.signal as signal

from bci.utils.bci_config import EEGConfig


class Filter:
    """
    Filter class for applying bandpass and notch filters to EEG data,
    supporting both offline (batch) and online (streaming) modes.

    Attributes
    ----------
    config : object
        Configuration object with filter parameters (order, frequencies, fs, channels, etc.)
    sos : np.ndarray
        Second-order sections representation of the filter.
    zi : np.ndarray
        Filter state for online filtering (initialized if online=True).
    """

    def __init__(
        self,
        config: EEGConfig,
        online: bool = False,
        n_channels_online: Optional[int] = None,
    ) -> None:
        """
        Initialize the Filter object.

        Parameters
        ----------
        config : object
            Configuration object with attributes:
                - order: int, filter order
                - frequencies: tuple/list, (low, high) cutoff frequencies
                - fs: float, sampling frequency
                - channels: list, channel names (for online mode)
        online : bool, optional
            If True, initializes filter state for online (streaming) filtering.
        n_channels_online : int, optional
            Number of channels for online filter state. If None, uses
            len(config.channels) - len(config.remove_channels).
        """
        self.config = config

        # Design a bandpass Butterworth filter using second-order sections (SOS)
        self.sos: np.ndarray = scipy.signal.butter(
            self.config.order,
            Wn=np.array(self.config.frequencies),
            btype="bandpass",
            fs=self.config.fs,
            output="sos",
        )

        # In online mode, maintain filter state for each channel
        if online:
            nr_of_channels = (
                n_channels_online
                if n_channels_online is not None
                else len(self.config.channels) - len(self.config.remove_channels)
            )
            # zi shape: (n_sections, 2)
            self.zi: np.ndarray = scipy.signal.sosfilt_zi(self.sos)
            # Expand to (n_sections, n_channels, 2)
            self.zi = np.repeat(self.zi[:, np.newaxis, :], nr_of_channels, axis=1)
            print(f"Initialized filter state with dimension {self.zi.shape}.")

    def apply_filter_offline(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply the designed SOS filter to the entire raw data (offline/batch).

        Parameters
        ----------
        data : np.ndarray
            Raw EEG data (n_channels, n_time)

        Returns
        -------
        filtered_data : np.ndarray
            Filtered data -> Suitable for the .apply_function of raw
        """
        # Use SOS for actual filtering (stability)
        filtered_signal: np.ndarray = scipy.signal.sosfilt(self.sos, signal, axis=-1)
        return filtered_signal

    def apply_filter_online(self, data_chunk: np.ndarray) -> np.ndarray:
        """
        Apply the filter to an incoming data chunk, maintaining state for online use.

        Parameters
        ----------
        data_chunk : np.ndarray
            EEG data chunk of shape (n_channels, n_samples)

        Returns
        -------
        filtered_chunk : np.ndarray
            Filtered data chunk of same shape as input
        """
        filtered_chunk = np.zeros_like(data_chunk)
        # Process each channel in parallel using vectorized operations
        filtered_chunk, self.zi = scipy.signal.sosfilt(
            self.sos, data_chunk, zi=self.zi, axis=-1
        )
        return filtered_chunk

    def apply_notch(self, raw: mne.io.BaseRaw, freqs: list) -> mne.io.BaseRaw:
        """
        Apply a notch filter to the raw data for each frequency in freqs.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG data object
        freqs : list
            List of frequencies (float) at which to apply the notch filter

        Returns
        -------
        raw : mne.io.BaseRaw
            Raw object with notch filtering applied
        """
        for freq_band in freqs:
            raw.notch_filter(freqs=freq_band)  # since Nyquist = 80 Hz  # europe = 50 Hz
        return raw

    def get_filter_latency(self) -> float:
        """
        Estimate the average group delay (latency) of the filter in milliseconds.

        Returns
        -------
        avg_latency_ms : float
            Average filter latency in milliseconds
        """
        # Convert SOS to transfer function
        b, a = scipy.signal.sos2tf(self.sos)

        # Compute group delay for each frequency
        w, gd = scipy.signal.group_delay((b, a))
        freqs = np.linspace(self.config.frequencies[0], self.config.frequencies[1], 100)
        gd_interp = np.interp(freqs, w * self.config.fs / (2 * np.pi), gd)

        # Average latency in milliseconds
        avg_latency_ms = np.mean(gd_interp) / self.config.fs * 1000

        # # Plot the group delay
        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(10, 6))
        # plt.plot(freqs, gd_interp, label="Group Delay")
        # plt.axvline(
        #     x=self.config.frequencies[0], color="r", linestyle="--", label="Low Cutoff"
        # )
        # plt.axvline(
        #     x=self.config.frequencies[1], color="g", linestyle="--", label="High Cutoff"
        # )
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Group Delay (samples)")
        # plt.title("Group Delay of Bandpass Filter")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        return avg_latency_ms


# change: helper to extract filter params safely from config objects or dicts
def get_filter_params_from_config(config):
    def _get(cfg, key, default=None):
        try:
            return getattr(cfg, key)
        except Exception:
            try:
                return cfg[key]
            except Exception:
                return default

    return {
        "fmin": _get(config, "fmin", 1.0),
        "fmax": _get(config, "fmax", 35.0),
        "reference": _get(config, "reference", "M1_M2"),
        "use_notch": _get(config, "use_notch", False),
        "notch_freq": _get(config, "notch_freq", 50.0),
        "fs": _get(config, "fs", _get(config, "sfreq", 250.0)),
        # change: include motor_channels so callers can filter/pick the same channels Daria uses
        "motor_channels": _get(config, "motor_channels", None),
    }


# change: factory to create the scipy-based Filter object from filtering.py using the same config
def create_filter_from_config(config, online: bool = False):
    # Local import to avoid circular imports at module import time
    from bci.preprocessing.filtering import Filter

    # Pass the config object directly — Filter expects config with attributes used in its ctor
    return Filter(config, online=online)


# change: approximate latency estimator for FIR filters used in offline pipeline
def estimate_filter_latency(
    fmin: float, fmax: float, fs: float, numtaps: int = 101
) -> float:
    # Ensure valid bounds
    if numtaps < 3:
        numtaps = 3

    # Design bandpass FIR (linear-phase) and compute group delay
    try:
        b = signal.firwin(numtaps, [fmin, fmax], pass_zero=False, fs=fs)
        w, gd = signal.group_delay((b, 1))
        # Average group delay in samples (focus on passband region)
        mean_gd = float(np.mean(gd))
        return mean_gd / fs * 1000.0
    except Exception:
        # Fallback: return conservative default (e.g., 10 ms)
        return 10.0


# change: convenience wrapper to filter train/test Raw objects together
def filter_dataset_pair(
    raw_train: mne.io.BaseRaw, raw_test: mne.io.BaseRaw, config
) -> Tuple[mne.io.BaseRaw, mne.io.BaseRaw, float]:
    params = get_filter_params_from_config(config)

    fmin = params["fmin"]
    fmax = params["fmax"]
    ref = params["reference"]
    use_notch = params["use_notch"]
    notch_freq = params["notch_freq"]
    fs = params.get("fs", 250.0)

    # Use the existing preprocessing.filter_raw function
    from bci.preprocessing.preprocessing import filter_raw

    filtered_train = filter_raw(
        raw=raw_train,
        fmin=fmin,
        fmax=fmax,
        reference=ref,
        use_notch=use_notch,
        notch_freq=notch_freq,
    )

    filtered_test = filter_raw(
        raw=raw_test,
        fmin=fmin,
        fmax=fmax,
        reference=ref,
        use_notch=use_notch,
        notch_freq=notch_freq,
    )

    # change: If the project config specifies motor_channels (Daria's selection),
    # change: pick those channels after filtering so the rest of preprocessing uses
    # change: the same channel subset. Picking after filtering preserves reference handling.
    motor_chs = params.get("motor_channels")
    if motor_chs is not None:
        try:
            # perform in-place pick on copies to avoid modifying original raws
            filtered_train = filtered_train.copy()
            filtered_train.pick(motor_chs)

            filtered_test = filtered_test.copy()
            filtered_test.pick(motor_chs)
        except Exception:
            # If pick fails (e.g., channel name mismatch), keep originals and warn
            import warnings

            warnings.warn(
                "Could not pick motor_channels from raw objects; leaving full channel set"
            )

    latency_ms = estimate_filter_latency(fmin, fmax, fs)

    return filtered_train, filtered_test, latency_ms

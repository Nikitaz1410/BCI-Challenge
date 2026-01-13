#change: New module extracted from preprocessing.py to centralize filter helpers
"""
Filter helpers: parameter parsing, factory, latency estimation, and dataset filtering helper.
This keeps filtering logic in a single file for clarity and reuse.
"""
from typing import Tuple

import numpy as np
import mne
import scipy.signal as signal


#change: helper to extract filter params safely from config objects or dicts
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
        #change: include motor_channels so callers can filter/pick the same channels Daria uses
        "motor_channels": _get(config, "motor_channels", None),
    }


#change: factory to create the scipy-based Filter object from filtering.py using the same config
def create_filter_from_config(config, online: bool = False):
    # Local import to avoid circular imports at module import time
    from bci.preprocessing.filtering import Filter

    # Pass the config object directly — Filter expects config with attributes used in its ctor
    return Filter(config, online=online)


#change: approximate latency estimator for FIR filters used in offline pipeline
def estimate_filter_latency(fmin: float, fmax: float, fs: float, numtaps: int = 101) -> float:
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


#change: convenience wrapper to filter train/test Raw objects together
def filter_dataset_pair(raw_train: mne.io.BaseRaw, raw_test: mne.io.BaseRaw, config) -> Tuple[mne.io.BaseRaw, mne.io.BaseRaw, float]:
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

    #change: If the project config specifies motor_channels (Daria's selection),
    #change: pick those channels after filtering so the rest of preprocessing uses
    #change: the same channel subset. Picking after filtering preserves reference handling.
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

            warnings.warn("Could not pick motor_channels from raw objects; leaving full channel set")

    latency_ms = estimate_filter_latency(fmin, fmax, fs)

    return filtered_train, filtered_test, latency_ms

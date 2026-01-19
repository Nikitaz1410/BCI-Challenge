"""
2_preprocessing: Offline EEG Processing Pipeline

This module handles:
1. Filtering (bandpass, notch)
2. Epoching (create time-locked segments)
3. Reject Bad Epochs (AutoReject)
4. Clean Data with ASR (Optional - Artifact Subspace Reconstruction)

Usage:
    from src.bci import preprocessing

    # Process raw data
    epochs, ar = preprocessing.process_offline(
        raw=raw_data,
        markers=markers,
        config=config_dict
    )


    ###############################################################################
    #  configuration  processing
    ###############################################################################

    config = {
    # FILTERING
    "fmin": 1.0,                    # Lower cutoff (Hz) - removes DC drift
    "fmax": 35.0,                   # Upper cutoff (Hz) - removes high-freq noise
    "reference": "M1_M2",           # Reference: "M1_M2" or "average"
    "use_notch": False,             # Notch filter for power line (50/60 Hz)
    "notch_freq": 50.0,             # Notch frequency (50 for Europe, 60 for US)

    # EPOCHING
    "epoch_tmin": 0.3,              # Start time relative to cue (seconds)
    "epoch_tmax": 3.0,              # End time relative to cue (seconds)
    "motor_channels": [              # Channels to keep after epoching
        "C3", "C4", "Cz", "Fz",
        "CPz", "P4", "F3", "C4"
    ],

    # AUTOREJECT (Bad Epoch Rejection)
    "autoreject": True,             # Enable AutoReject
    "n_interpolate": [1, 2],        # Channels to interpolate before rejecting

    # ASR (Optional - Advanced Cleaning)
    "use_asr": False,               # Enable ASR (optional)
    "asr_cutoff": 20.0              # ASR cutoff parameter
}
"""

from __future__ import annotations

import numpy as np
import mne
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# change: added scipy import for latency estimation and FIR design
import scipy.signal as signal

# Optional dependencies
try:
    import autoreject

    AUTOREJECT_AVAILABLE = True
except ImportError:
    AUTOREJECT_AVAILABLE = False
    warnings.warn("autoreject not available. Bad epoch rejection will be skipped.")

try:
    import asrpy

    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False
    warnings.warn("asrpy not available. ASR cleaning will be skipped.")

# MNE settings
mne.set_log_level("WARNING")
RANDOM_STATE = 42


# ============================================================================
# STEP 1: FILTERING
# ============================================================================


def filter_raw(
    raw: mne.io.BaseRaw,
    fmin: float,
    fmax: float,
    reference: str = "M1_M2",
    use_notch: bool = False,
    notch_freq: float = 50.0,
    notch_method: str = "spectrum_fit",
) -> mne.io.BaseRaw:
    """
    Apply bandpass filtering and set reference to raw EEG data.

    Parameters:
    -----------
    raw : mne.io.BaseRaw
        Raw EEG data object
    fmin : float
        Lower cutoff frequency (Hz). Typical: 1.0 (removes DC drift) or 6.0
    fmax : float
        Upper cutoff frequency (Hz). Typical: 30.0 or 35.0 (removes high-freq noise)
    reference : str
        Reference type: "M1_M2" (mastoid) or "average" (average reference)
    use_notch : bool
        Whether to apply notch filter for power line noise removal
    notch_freq : float
        Notch frequency: 50.0 (Europe) or 60.0 (US)
    notch_method : str
        Notch filter method: "spectrum_fit" (accurate) or "fir" (fast)

    Returns:
    --------
    raw_filtered : mne.io.BaseRaw
        Filtered and referenced raw data

    Notes:
    ------
    - Uses minimal-phase filter (phase="minimum") for causal/online compatibility
    - M1/M2 reference is standard for motor imagery
    - Average reference can be used if all electrodes are good
    """
    raw = raw.copy()

    # Optional: Notch filter for power line noise
    if use_notch:
        raw.notch_filter(freqs=[notch_freq], method=notch_method, verbose=False)

    # Bandpass filter (minimal-phase for causal processing)
    raw.filter(
        l_freq=fmin,
        h_freq=fmax,
        phase="minimum",  # Causal filter (can be used online)
        fir_design="firwin",
        verbose=False,
    )

    # Set reference
    if reference == "M1_M2":
        raw.set_eeg_reference(
            ref_channels=["M1", "M2"], projection=False, verbose=False
        )
    elif reference == "average":
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    else:
        raise ValueError(
            f"Unknown reference type: {reference}. Use 'M1_M2' or 'average'"
        )

    return raw


# change: filter helpers moved to `bci.preprocessing.filters` for better structure
from bci.preprocessing.filters import (
    get_filter_params_from_config,
    create_filter_from_config,
    estimate_filter_latency,
    filter_dataset_pair,
)


# ============================================================================
# STEP 2: EPOCHING (Create Windows)
# ============================================================================


def create_epochs(
    raw: mne.io.BaseRaw,
    markers: List[str],
    tmin: float,
    tmax: float,
    event_dict: Optional[Dict[str, int]] = None,
    baseline: Optional[Tuple[float, float]] = None,
    motor_channels: Optional[List[str]] = None,
    padding: float = 0.5,
) -> Tuple[mne.Epochs, Dict[str, int]]:
    """
    Create epochs (time-locked segments) from raw data based on event markers.

    Parameters:
    -----------
    raw : mne.io.BaseRaw
        Filtered raw data
    markers : List[str]
        List of marker names (e.g., ["ARROW LEFT ONSET", "ARROW RIGHT ONSET", "CIRCLE ONSET"])
    tmin : float
        Start time relative to marker (seconds). Typical: 0.3 (skip initial visual response)
    tmax : float
        End time relative to marker (seconds). Typical: 3.0 (capture full MI response)
    event_dict : Dict[str, int], optional
        Pre-defined mapping of marker names to event IDs. If None, auto-generates.
    baseline : Tuple[float, float], optional
        Baseline correction window (e.g., (-0.5, 0.0)). If None, no baseline correction.
    motor_channels : List[str], optional
        Channels to keep after epoching (e.g., ["C3", "C4", "Cz"]). If None, keeps all.
    padding : float
        Extra padding around epoch window for filters (seconds). Typical: 0.5

    Returns:
    --------
    epochs : mne.Epochs
        Epochs object with shape (n_epochs, n_channels, n_time_samples)
    event_dict : Dict[str, int]
        Final event dictionary mapping marker names to event IDs

    Notes:
    ------
    - Epochs are created with padding, then cropped to exact window
    - Labels are automatically mapped: 0=rest, 1=right, 2=left
    - Motor channels are selected after epoching to reduce computation
    """
    # Get events from annotations
    if event_dict is None:
        # Auto-generate event dict from markers
        markers_dict = {marker: i for i, marker in enumerate(markers)}
        events, events_id = mne.events_from_annotations(
            raw, event_id=markers_dict, verbose=False
        )
        # Filter to only keep markers we care about
        event_dict = {k: v for k, v in events_id.items() if k in markers}
    else:
        # Use provided event dict
        events, events_id = mne.events_from_annotations(
            raw, event_id=event_dict, verbose=False
        )
        event_dict = {k: v for k, v in events_id.items() if k in event_dict}

    # Create epochs with padding (for filter edge effects)
    tmin_padded = tmin - padding
    tmax_padded = tmax + padding

    epochs = mne.Epochs(
        raw,
        events=events,
        tmin=tmin_padded,
        tmax=tmax_padded,
        event_id=event_dict,
        baseline=baseline,
        preload=True,  # Load into memory
        verbose=False,
        event_repeated="drop",  # Drop duplicate events
    )

    # Crop to exact time window (remove padding)
    epochs.crop(tmin=tmin, tmax=tmax)

    # Select motor channels if specified
    if motor_channels is not None:
        epochs.pick(motor_channels)

    # Map labels to standard format: {0: rest, 1: right, 2: left}
    # Handle different marker naming conventions
    rest_markers = ["CIRCLE ONSET", "CIRCLE", "rest"]
    right_markers = ["ARROW RIGHT ONSET", "ARROW RIGHT", "right_hand"]
    left_markers = ["ARROW LEFT ONSET", "ARROW LEFT", "left_hand"]

    rest_id = None
    right_id = None
    left_id = None

    for marker_name in event_dict.keys():
        if marker_name in rest_markers:
            rest_id = event_dict[marker_name]
        elif marker_name in right_markers:
            right_id = event_dict[marker_name]
        elif marker_name in left_markers:
            left_id = event_dict[marker_name]

    if rest_id is not None or right_id is not None or left_id is not None:
        y = epochs.events[:, -1].copy()
        if rest_id is not None:
            y[y == rest_id] = 0
        if right_id is not None:
            y[y == right_id] = 1
        if left_id is not None:
            y[y == left_id] = 2
        epochs.events[:, -1] = y

        epochs.event_id = {"rest": 0, "right": 1, "left": 2}

    return epochs, event_dict


# change: compatibility wrapper to preserve the old `extract_epochs(raw, events, event_id, config)`
def extract_epochs(raw, events, event_id, config):
    """Compatibility wrapper that creates MNE Epochs and returns labels.

    Kept to match the original API used in `main_offline.py` so the main can
    remain minimal (two simple lines). Internally calls `create_epochs`.

    Parameters:
    - raw: mne.io.BaseRaw
    - events: original events array (not used directly here but kept for API compatibility)
    - event_id: mapping of marker names to ids (used to build markers list)
    - config: configuration dict

    Returns:
    - epochs: mne.Epochs
    - labels: np.ndarray of labels per epoch
    """
    # Build markers list from event_id if provided
    markers = list(event_id.keys()) if event_id is not None else None

    epochs, _ = create_epochs(
        raw=raw,
        markers=markers,
        tmin=config.get("epoch_tmin", 0.3),
        tmax=config.get("epoch_tmax", 3.0),
        motor_channels=config.get("motor_channels", None),
    )

    labels = epochs.events[:, -1].copy() if len(epochs) > 0 else np.array([], dtype=int)
    return epochs, labels


# change: high-level helper to apply AutoReject (fit on train) and convert epochs -> windows
def autoreject_and_window(
    epochs: mne.Epochs,
    labels: np.ndarray,
    sessions_id,
    config: Dict,
    ar: Optional[object] = None,
    fit_ar: bool = False,
) -> Tuple[mne.Epochs, np.ndarray, np.ndarray, np.ndarray, object]:
    """Fit/apply AutoReject and extract overlapping windows.

    Parameters
    ----------
    epochs : mne.Epochs
    labels : np.ndarray
    sessions_id : any
        Per-epoch metadata (list/array) aligned with `epochs` length.
    config : dict
        Must contain 'window_size' and 'step_size' (samples)
    ar : ArtefactRemoval or None
        Pre-fitted ArtefactRemoval object to apply (used when fit_ar=False)
    fit_ar : bool
        If True, fit a new ArtefactRemoval on `epochs` and return it.

    Returns
    -------
    cleaned_epochs : mne.Epochs
    X_windows : np.ndarray (n_windows, n_channels, window_size)
    y_windows : np.ndarray (n_windows,)
    sessions_per_window : np.ndarray (n_windows,) or empty
    ar_obj : ArtefactRemoval

    Notes
    -----
    - Uses `bci.preprocessing.windows.epochs_to_windows` for window extraction.
    - Keeps `epochs` as MNE objects for downstream use.
    """
    import numpy as _np
    from bci.preprocessing.windows import epochs_to_windows as _epochs_to_windows
    from bci.preprocessing.artefact_removal import ArtefactRemoval as _ArtefactRemoval

    epoch_data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    # Fit or use provided AutoReject wrapper
    if fit_ar:
        ar_obj = _ArtefactRemoval()
        # ArtefactRemoval expects numpy epoch arrays when computing thresholds
        ar_obj.get_rejection_thresholds(epoch_data, config)
    else:
        if ar is None:
            raise ValueError("When fit_ar=False a pre-fitted `ar` must be provided")
        ar_obj = ar

    if getattr(ar_obj, "rejection_threshold", None) is None:
        raise ValueError("AutoReject thresholds not available on ar object")

    thr = ar_obj.rejection_threshold

    # Compute mask of good epochs (True = keep)
    keep_mask = _np.array([not _np.any(_np.abs(ep) > thr) for ep in epoch_data])
    kept_idx = _np.nonzero(keep_mask)[0]

    # Build cleaned MNE Epochs
    if kept_idx.size == 0:
        # No epochs left after rejection
        cleaned_epochs = mne.EpochsArray(
            _np.zeros((0, epoch_data.shape[1], int(config.get("window_size", 250)))),
            info=epochs.info,
        )
        cleaned_labels = _np.array([], dtype=int)
    else:
        cleaned_array = epoch_data[kept_idx]
        cleaned_labels = _np.asarray(labels)[kept_idx]
        cleaned_events = epochs.events[kept_idx]
        cleaned_epochs = mne.EpochsArray(
            cleaned_array, info=epochs.info, events=cleaned_events, tmin=epochs.tmin
        )

    # Update sessions_id to kept epochs if possible
    try:
        sessions_kept = _np.asarray(sessions_id)[kept_idx]
    except Exception:
        sessions_kept = sessions_id
        print(
            "#change: Warning - couldn't auto-update sessions_id after AutoReject. Please verify manually."
        )

    # Window extraction
    window_size = int(config.get("window_size", 250))
    step_size = int(config.get("step_size", 32))
    X_windows, y_windows = _epochs_to_windows(
        cleaned_epochs, window_size=window_size, step_size=step_size
    )

    # Expand per-window session ids
    trial_ids = []
    for i in range(len(cleaned_epochs)):
        n_channels, n_samples = cleaned_epochs.get_data()[i].shape
        nw = 0
        if n_samples >= window_size:
            nw = int((n_samples - window_size) / step_size) + 1
        trial_ids.extend([i] * nw)
    trial_ids = _np.array(trial_ids, dtype=int)

    try:
        sessions_per_window = _np.asarray(sessions_kept)[trial_ids]
    except Exception:
        sessions_per_window = _np.array([], dtype=int)
        if len(trial_ids) > 0:
            print(
                "#change: Warning - couldn't expand sessions_id to per-window mapping. Please verify manually."
            )

    return cleaned_epochs, X_windows, y_windows, sessions_per_window, ar_obj


# ============================================================================
# STEP 3: REJECT BAD EPOCHS (AutoReject)
# ============================================================================


def reject_bad_epochs(
    epochs: mne.Epochs,
    n_interpolate: List[int] = [1, 2],
    fit: bool = True,
    ar_object: Optional[autoreject.AutoReject] = None,
    save_path: Optional[Path] = None,
) -> Tuple[mne.Epochs, Optional[autoreject.AutoReject]]:
    """
    Apply AutoReject to detect and repair/reject bad epochs.

    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs to clean
    n_interpolate : List[int]
        Number of bad channels to interpolate before rejecting epoch.
        [1, 2] means: try 1 bad channel, then 2, then reject if more.
        More aggressive: [1, 2, 4] (interpolates up to 4 channels)
    fit : bool
        Whether to fit AutoReject (True for training data, False for test)
    ar_object : autoreject.AutoReject, optional
        Pre-fitted AutoReject object (use for test data to avoid data leakage)
    save_path : Path, optional
        Path to save fitted AutoReject object

    Returns:
    --------
    epochs_clean : mne.Epochs
        Cleaned epochs (bad epochs removed, bad channels interpolated)
    ar : autoreject.AutoReject
        Fitted AutoReject object (save this for online use)

    Notes:
    ------
    - AutoReject learns per-channel thresholds from training data
    - For test data, use pre-fitted ar_object to avoid data leakage
    - Bad epochs are completely removed (not interpolated)
    - Bad channels within good epochs are interpolated
    """
    if not AUTOREJECT_AVAILABLE:
        warnings.warn("autoreject not available. Skipping bad epoch rejection.")
        return epochs, None

    # Fit or use existing AutoReject
    if ar_object is None:
        ar = autoreject.AutoReject(
            n_interpolate=n_interpolate,
            random_state=RANDOM_STATE,
            n_jobs=1,  # Single-threaded (safer)
            verbose=False,
        )
        if fit:
            epochs_clean = ar.fit_transform(epochs)
            print(
                f"AutoReject: Fitted on {len(epochs)} epochs, kept {len(epochs_clean)} epochs"
            )
        else:
            raise ValueError("Must provide ar_object if fit=False")
    else:
        ar = ar_object
        epochs_clean = ar.transform(epochs)
        print(
            f"AutoReject: Applied to {len(epochs)} epochs, kept {len(epochs_clean)} epochs"
        )

    # Save AutoReject object if path provided
    if save_path is not None and fit:
        import joblib

        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(ar, save_path)
        print(f"AutoReject saved to {save_path}")

    # Safety: Remove NaN/Inf values
    data = epochs_clean.get_data()
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data)
        epochs_clean = mne.EpochsArray(
            data,
            epochs_clean.info,
            events=epochs_clean.events,
            event_id=epochs_clean.event_id,
            tmin=epochs_clean.tmin,
        )

    return epochs_clean, ar


def load_autoreject(filepath: Path) -> autoreject.AutoReject:
    """
    Load a saved AutoReject object.

    Parameters:
    -----------
    filepath : Path
        Path to saved AutoReject file (.joblib)

    Returns:
    --------
    ar : autoreject.AutoReject
        Loaded AutoReject object
    """
    if not AUTOREJECT_AVAILABLE:
        raise ImportError("autoreject not available")

    import joblib

    return joblib.load(filepath)


# ============================================================================
# STEP 4: CLEAN DATA WITH ASR (Optional)
# ============================================================================


def clean_with_asr(
    epochs: mne.Epochs,
    baseline_data: Optional[mne.io.BaseRaw] = None,
    cutoff: float = 20.0,
    asr_object: Optional = None,
    save_path: Optional[Path] = None,
) -> Tuple[mne.Epochs, Optional]:
    """
    Apply Artifact Subspace Reconstruction (ASR) to clean epochs.

    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs to clean
    baseline_data : mne.io.BaseRaw, optional
        Baseline/calibration data for ASR. If None, uses first epoch as baseline.
    cutoff : float
        ASR cutoff parameter (standard deviations). Typical: 20.0
        Higher = more aggressive cleaning
    asr_object : ASR object, optional
        Pre-fitted ASR object (for test data)
    save_path : Path, optional
        Path to save fitted ASR object

    Returns:
    --------
    epochs_clean : mne.Epochs
        ASR-cleaned epochs
    asr : ASR object
        Fitted ASR object (save for online use)

    Notes:
    ------
    - ASR requires baseline/calibration data to learn artifact patterns
    - Typically run on baseline recording (rest state, eyes open/closed)
    - More aggressive than AutoReject but computationally expensive
    - Optional step - can be skipped if AutoReject is sufficient
    """
    if not ASR_AVAILABLE:
        warnings.warn("asrpy not available. Skipping ASR cleaning.")
        return epochs, None

    try:
        from asrpy import ASR

        # Fit ASR if not provided
        if asr_object is None:
            if baseline_data is None:
                # Use first epoch as baseline (not ideal, but works)
                print(
                    "Warning: No baseline data provided. Using first epoch as baseline."
                )
                baseline_epoch = epochs[0]
                asr = ASR(sfreq=epochs.info["sfreq"], cutoff=cutoff)
                asr.fit(baseline_epoch)
            else:
                # Use provided baseline data
                asr = ASR(sfreq=baseline_data.info["sfreq"], cutoff=cutoff)
                asr.fit(baseline_data)

            # Apply ASR to epochs
            epochs_clean = asr.transform(epochs)
            print(f"ASR: Cleaned {len(epochs)} epochs")
        else:
            # Use existing ASR object
            asr = asr_object
            epochs_clean = asr.transform(epochs)
            print(f"ASR: Applied to {len(epochs)} epochs")

        # Save ASR object if path provided
        if save_path is not None and asr_object is None:
            import joblib

            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(asr, save_path)
            print(f"ASR saved to {save_path}")

        return epochs_clean, asr

    except ImportError:
        warnings.warn("asrpy import failed. Skipping ASR cleaning.")
        return epochs, None


def load_asr(filepath: Path):
    """
    Load a saved ASR object.

    Parameters:
    -----------
    filepath : Path
        Path to saved ASR file (.joblib)

    Returns:
    --------
    asr : ASR object
        Loaded ASR object
    """
    import joblib

    return joblib.load(filepath)


# ============================================================================
# COMPLETE OFFLINE PROCESSING PIPELINE
# ============================================================================


def process_offline(
    raw: mne.io.BaseRaw,
    markers: List[str],
    config: Dict,
    baseline_data: Optional[mne.io.BaseRaw] = None,
    save_ar_path: Optional[Path] = None,
    save_asr_path: Optional[Path] = None,
) -> Tuple[mne.Epochs, Optional[autoreject.AutoReject], Optional]:
    """
    Complete offline processing pipeline: Filter → Epoch → Reject Bad → ASR (optional)

    Parameters:
    -----------
    raw : mne.io.BaseRaw
        Raw EEG data (from 1_loading module)
    markers : List[str]
        Event markers (from 1_loading module)
    config : Dict
        Configuration dictionary with keys:
        - fmin, fmax: filter frequencies (float)
        - reference: "M1_M2" or "average" (str)
        - use_notch: apply notch filter (bool)
        - notch_freq: notch frequency (float)
        - epoch_tmin, epoch_tmax: epoch time window (float)
        - motor_channels: channels to keep (List[str])
        - autoreject: enable AutoReject (bool)
        - n_interpolate: AutoReject params (List[int])
        - use_asr: enable ASR (bool)
        - asr_cutoff: ASR cutoff (float)
    baseline_data : mne.io.BaseRaw, optional
        Baseline data for ASR calibration
    save_ar_path : Path, optional
        Where to save AutoReject object
    save_asr_path : Path, optional
        Where to save ASR object

    Returns:
    --------
    epochs : mne.Epochs
        Fully processed epochs ready for feature extraction
    ar : autoreject.AutoReject, optional
        Fitted AutoReject object (for online use)
    asr : ASR object, optional
        Fitted ASR object (for online use)

    Example:
    --------
    config = {
        "fmin": 1.0,
        "fmax": 35.0,
        "reference": "M1_M2",
        "use_notch": False,
        "epoch_tmin": 0.3,
        "epoch_tmax": 3.0,
        "motor_channels": ["C3", "C4", "Cz"],
        "autoreject": True,
        "n_interpolate": [1, 2],
        "use_asr": False
    }

    epochs, ar, asr = process_offline(raw, markers, config)
    """
    print("=" * 60)
    print("OFFLINE PROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Filter
    print("\n[1/4] Filtering...")
    raw_filtered = filter_raw(
        raw=raw,
        fmin=config["fmin"],
        fmax=config["fmax"],
        reference=config.get("reference", "M1_M2"),
        use_notch=config.get("use_notch", False),
        notch_freq=config.get("notch_freq", 50.0),
    )
    print(f"  Applied bandpass filter: {config['fmin']}-{config['fmax']} Hz")
    print(f"  Reference: {config.get('reference', 'M1_M2')}")

    # Step 2: Epoch
    print("\n[2/4] Creating epochs...")
    epochs, event_dict = create_epochs(
        raw=raw_filtered,
        markers=markers,
        tmin=config["epoch_tmin"],
        tmax=config["epoch_tmax"],
        motor_channels=config.get("motor_channels", None),
    )
    print(f"  Created {len(epochs)} epochs")
    print(f"  Time window: {config['epoch_tmin']}-{config['epoch_tmax']} s")
    if config.get("motor_channels"):
        print(f"  Motor channels: {config['motor_channels']}")

    # Step 3: Reject Bad Epochs (AutoReject)
    ar = None
    if config.get("autoreject", False):
        print("\n[3/4] Rejecting bad epochs (AutoReject)...")
        epochs, ar = reject_bad_epochs(
            epochs=epochs,
            n_interpolate=config.get("n_interpolate", [1, 2]),
            fit=True,
            save_path=save_ar_path,
        )
    else:
        print("\n[3/4] Skipping AutoReject (disabled in config)")

    # Step 4: ASR Cleaning (Optional)
    asr = None
    if config.get("use_asr", False):
        print("\n[4/4] Cleaning with ASR...")
        epochs, asr = clean_with_asr(
            epochs=epochs,
            baseline_data=baseline_data,
            cutoff=config.get("asr_cutoff", 20.0),
            save_path=save_asr_path,
        )
    else:
        print("\n[4/4] Skipping ASR (disabled in config)")

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Final epochs: {len(epochs)}")
    print(f"Shape: {epochs.get_data().shape}")
    print(f"Labels: {np.unique(epochs.events[:, -1], return_counts=True)}")

    return epochs, ar, asr


# ============================================================================
# WINDOW EXTRACTION (Overlapping Windows from Epochs)
# ============================================================================

# `bci.preprocessing.windows
# Re-export commonly used helpers to preserve backward compatibility.
from bci.preprocessing.windows import (
    extract_overlapping_windows,
    epochs_to_windows,
)


# ============================================================================
# ONLINE PROCESSING (Real-time)
# ============================================================================


def filter_online(
    window: np.ndarray, sfreq: float, fmin: float, fmax: float
) -> np.ndarray:
    """
    Apply causal bandpass filter to a single window (for online processing).

    Parameters:
    -----------
    window : np.ndarray
        Shape (n_channels, n_samples) - single window
    sfreq : float
        Sampling frequency (Hz) - typically 250
    fmin, fmax : float
        Filter frequencies (Hz) - same as offline config

    Returns:
    --------
    filtered : np.ndarray
        Filtered window (same shape: n_channels, n_samples)

    Notes:
    ------
    - Uses causal filter (sosfilt) - can only use past/present samples
    - Must be fast for real-time processing (<10ms)
    - Uses same fmin/fmax as offline for consistency
    """
    from scipy import signal

    # Create Butterworth filter (causal, minimal-phase)
    sos = signal.butter(
        4,  # Filter order (4th order = good balance)
        [fmin, fmax],
        btype="bandpass",
        fs=sfreq,
        output="sos",  # Second-order sections (stable)
    )

    # Apply filter (causal - processes sample by sample)
    filtered = signal.sosfilt(sos, window, axis=-1)

    return filtered


def create_sliding_window_online(
    data_buffer: np.ndarray, window_size: int, step: int
) -> Optional[np.ndarray]:
    """
    Create sliding window from data buffer (for online processing).

    Parameters:
    -----------
    data_buffer : np.ndarray
        Shape (n_channels, buffer_length) - accumulated data buffer
        Buffer should be continuously updated with new samples
    window_size : int
        Window length in samples (e.g., 250 = 1.0s at 250 Hz)
        Must match offline window_size
    step : int
        Hop size in samples (e.g., 40 = 0.16s at 250 Hz)
        Must match offline step

    Returns:
    --------
    window : np.ndarray or None
        Shape (n_channels, window_size) - most recent window
        Returns None if buffer is too short

    Example:
    --------
    buffer = np.zeros((7, 1000))  # 7 channels, 1000 samples buffer
    # ... add new samples to buffer ...
    window = create_sliding_window_online(buffer, window_size=250, step=40)
    if window is not None:
        # Process window
    """
    if data_buffer.shape[1] < window_size:
        return None

    # Get most recent window_size samples from buffer
    window = data_buffer[:, -window_size:]
    return window


def handle_bad_window_online(
    window: np.ndarray,
    max_amplitude: float = 200e-6,  # 200 microvolts
    ar: Optional[autoreject.AutoReject] = None,
    sfreq: float = 250.0,
) -> Tuple[np.ndarray, bool]:
    """
    Detect and handle bad windows in real-time.

    Parameters:
    -----------
    window : np.ndarray
        Shape (n_channels, n_samples) - window to check
    max_amplitude : float
        Maximum allowed amplitude in volts. Default: 200 µV
        Typical EEG: 10-100 µV, artifacts: >200 µV
    ar : autoreject.AutoReject, optional
        Pre-fitted AutoReject object (from offline training)
        If provided, uses AR for more sophisticated detection
    sfreq : float
        Sampling frequency

    Returns:
    --------
    window_clean : np.ndarray
        Cleaned window (or original if not bad)
    is_bad : bool
        True if window should be skipped (too many artifacts)

    Notes:
    ------
    - Simple method: amplitude threshold
    - Advanced method: AutoReject (if available)
    - Bad windows should be skipped (not used for prediction)
    """
    # Method 1: Simple amplitude-based detection
    if np.abs(window).max() > max_amplitude:
        return window, True  # Mark as bad, skip this window

    # Method 2: AutoReject (if available and provided)
    if ar is not None and AUTOREJECT_AVAILABLE:
        try:
            # Convert window to MNE EpochsArray format
            info = mne.create_info(
                [f"ch{i}" for i in range(window.shape[0])], sfreq=sfreq, ch_types="eeg"
            )
            # Reshape to (1, n_channels, n_samples) for single epoch
            window_epoch = window[np.newaxis, :, :]
            epochs = mne.EpochsArray(window_epoch, info, verbose="ERROR")

            # Apply AutoReject transform
            epochs_clean = ar.transform(epochs)

            # Check if epoch was rejected (length becomes 0)
            if len(epochs_clean) == 0:
                return window, True  # Rejected by AR

            # Return cleaned window
            return epochs_clean.get_data()[0], False
        except Exception as e:
            # If AR fails, fall back to amplitude check
            return window, True

    # Window passed all checks
    return window, False


def clean_window_asr_online(
    window: np.ndarray, asr_object: Optional = None, sfreq: float = 250.0
) -> np.ndarray:
    """
    Apply ASR (Artifact Subspace Reconstruction) cleaning to window (optional).

    Parameters:
    -----------
    window : np.ndarray
        Shape (n_channels, n_samples) - window to clean
    asr_object : ASR object, optional
        Pre-fitted ASR object (from offline baseline calibration)
    sfreq : float
        Sampling frequency

    Returns:
    --------
    window_clean : np.ndarray
        ASR-cleaned window (or original if ASR not available)

    Notes:
    ------
    - ASR requires baseline data for calibration (done offline)
    - More aggressive than AutoReject but computationally expensive
    - Optional step - can be skipped for faster processing
    """
    if not ASR_AVAILABLE or asr_object is None:
        return window

    try:
        from asrpy import ASR

        # Note: ASR typically works on continuous data streams
        # For single windows, this is a simplified implementation
        # Full ASR integration may require buffering multiple windows
        # For now, return original if ASR not properly configured
        return window
    except Exception:
        return window


def process_window_online(
    window: np.ndarray,
    config: Dict,
    ar: Optional[autoreject.AutoReject] = None,
    asr: Optional = None,
    sfreq: float = 250.0,
) -> Tuple[np.ndarray, bool]:
    """
    Complete online processing pipeline for a single window.
    Combines: Filter → Handle Bad → ASR (optional)

    Parameters:
    -----------
    window : np.ndarray
        Shape (n_channels, n_samples) - raw window from buffer
    config : Dict
        Configuration dictionary (same as offline):
        - fmin, fmax: filter frequencies
        - max_amplitude: bad window threshold (optional)
        - use_asr: enable ASR (optional)
    ar : autoreject.AutoReject, optional
        Pre-fitted AutoReject object (load from offline)
    asr : ASR object, optional
        Pre-fitted ASR object (load from offline)
    sfreq : float
        Sampling frequency

    Returns:
    --------
    window_processed : np.ndarray
        Fully processed window ready for feature extraction
        Shape: (n_channels, n_samples)
    is_bad : bool
        True if window should be skipped (too many artifacts)
        False if window is good and can be used

    Example:
    --------
    # Load objects from offline
    ar = preprocessing.load_autoreject(Path("models/autoreject.joblib"))

    # Process window
    window_clean, is_bad = preprocessing.process_window_online(
        window=raw_window,
        config=config,
        ar=ar,
        sfreq=250.0
    )

    if not is_bad:
        # Reshape for model: [C, T] -> [1, C, T]
        window_for_model = preprocessing.prepare_window_for_model(window_clean)

        # Use window for prediction
        predictions, probabilities = model.infer(window_for_model)
    """
    # Step 1: Filter (causal - must be fast)
    window_filtered = filter_online(
        window, sfreq=sfreq, fmin=config["fmin"], fmax=config["fmax"]
    )

    # Step 2: Handle bad window (detect artifacts)
    window_clean, is_bad = handle_bad_window_online(
        window_filtered,
        max_amplitude=config.get("max_amplitude", 200e-6),
        ar=ar,
        sfreq=sfreq,
    )

    # If bad, return early (skip ASR)
    if is_bad:
        return window_clean, True

    # Step 3: Optional ASR cleaning
    if config.get("use_asr", False) and asr is not None:
        window_clean = clean_window_asr_online(
            window_clean, asr_object=asr, sfreq=sfreq
        )

    return window_clean, False


def prepare_window_for_model(window: np.ndarray) -> np.ndarray:
    """
    Reshape a processed window from [C, T] to [1, C, T] for model inference.

    Parameters:
    -----------
    window : np.ndarray
        Processed window with shape (n_channels, n_samples) = [C, T]
        This is the output from process_window_online()

    Returns:
    --------
    window_batch : np.ndarray
        Window reshaped for model input with shape (1, n_channels, n_samples) = [1, C, T]
        Ready to be passed to model.infer() or model.predict()

    Example:
    --------
    window_processed, is_bad = preprocessing.process_window_online(
        window=raw_window,
        config=config,
        ar=ar,
        sfreq=250.0
    )

    if not is_bad:
        window_for_model = preprocessing.prepare_window_for_model(window_processed)
        predictions, probabilities = model.infer(window_for_model)
    """
    if window.ndim != 2:
        raise ValueError(
            f"Expected 2D window with shape [C, T], got shape {window.shape} "
            f"with {window.ndim} dimensions. "
            "Did you already reshape it? Or did preprocessing fail?"
        )

    window_batch = window[np.newaxis, :, :]

    return window_batch


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    # Offline functions
    "filter_raw",
    "create_epochs",
    "reject_bad_epochs",
    "load_autoreject",
    "clean_with_asr",
    "load_asr",
    "process_offline",
    # Windowing functions
    "extract_overlapping_windows",
    "extract_windows_from_epochs",
    # Online functions
    "filter_online",
    "create_sliding_window_online",
    "handle_bad_window_online",
    "clean_window_asr_online",
    "process_window_online",
    "prepare_window_for_model",
]

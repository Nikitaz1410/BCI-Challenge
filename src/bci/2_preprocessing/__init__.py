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
    notch_method: str = "spectrum_fit"
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
        raw.notch_filter(
            freqs=[notch_freq],
            method=notch_method,
            verbose=False
        )
    
    # Bandpass filter (minimal-phase for causal processing)
    raw.filter(
        l_freq=fmin,
        h_freq=fmax,
        phase="minimum",  # Causal filter (can be used online)
        fir_design="firwin",
        verbose=False
    )
    
    # Set reference
    if reference == "M1_M2":
        raw.set_eeg_reference(
            ref_channels=["M1", "M2"],
            projection=False,
            verbose=False
        )
    elif reference == "average":
        raw.set_eeg_reference(
            ref_channels="average",
            projection=False,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown reference type: {reference}. Use 'M1_M2' or 'average'")
    
    return raw


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
    padding: float = 0.5
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
    if "CIRCLE ONSET" in event_dict:
        rest_id = event_dict["CIRCLE ONSET"]
        right_id = event_dict.get("ARROW RIGHT ONSET", None)
        left_id = event_dict.get("ARROW LEFT ONSET", None)
        
        y = epochs.events[:, -1].copy()
        y[y == rest_id] = 0  # Rest
        if right_id is not None:
            y[y == right_id] = 1  # Right hand
        if left_id is not None:
            y[y == left_id] = 2  # Left hand
        epochs.events[:, -1] = y
        
        # Update event_id for consistency
        epochs.event_id = {"rest": 0, "right": 1, "left": 2}
    
    return epochs, event_dict


# ============================================================================
# STEP 3: REJECT BAD EPOCHS (AutoReject)
# ============================================================================

def reject_bad_epochs(
    epochs: mne.Epochs,
    n_interpolate: List[int] = [1, 2],
    fit: bool = True,
    ar_object: Optional[autoreject.AutoReject] = None,
    save_path: Optional[Path] = None
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
            verbose=False
        )
        if fit:
            epochs_clean = ar.fit_transform(epochs)
            print(f"AutoReject: Fitted on {len(epochs)} epochs, kept {len(epochs_clean)} epochs")
        else:
            raise ValueError("Must provide ar_object if fit=False")
    else:
        ar = ar_object
        epochs_clean = ar.transform(epochs)
        print(f"AutoReject: Applied to {len(epochs)} epochs, kept {len(epochs_clean)} epochs")
    
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
            tmin=epochs_clean.tmin
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
    save_path: Optional[Path] = None
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
                print("Warning: No baseline data provided. Using first epoch as baseline.")
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
    save_asr_path: Optional[Path] = None
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
        notch_freq=config.get("notch_freq", 50.0)
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
        motor_channels=config.get("motor_channels", None)
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
            save_path=save_ar_path
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
            save_path=save_asr_path
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
# EXPORTS
# ============================================================================

__all__ = [
    "filter_raw",
    "create_epochs",
    "reject_bad_epochs",
    "load_autoreject",
    "clean_with_asr",
    "load_asr",
    "process_offline",
]


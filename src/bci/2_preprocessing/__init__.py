from __future__ import annotations

import numpy as np
import mne
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

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

mne.set_log_level("WARNING")
RANDOM_STATE = 42


def filter_raw(
    raw: mne.io.BaseRaw,
    fmin: float,
    fmax: float,
    reference: str = "M1_M2",
    use_notch: bool = False,
    notch_freq: float = 50.0,
    notch_method: str = "spectrum_fit"
) -> mne.io.BaseRaw:
    raw = raw.copy()
    
    if use_notch:
        raw.notch_filter(
            freqs=[notch_freq],
            method=notch_method,
            verbose=False
        )
    
    raw.filter(
        l_freq=fmin,
        h_freq=fmax,
        phase="minimum",
        fir_design="firwin",
        verbose=False
    )
    
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
    if event_dict is None:
        markers_dict = {marker: i for i, marker in enumerate(markers)}
        events, events_id = mne.events_from_annotations(
            raw, event_id=markers_dict, verbose=False
        )
        event_dict = {k: v for k, v in events_id.items() if k in markers}
    else:
        events, events_id = mne.events_from_annotations(
            raw, event_id=event_dict, verbose=False
        )
        event_dict = {k: v for k, v in events_id.items() if k in event_dict}
    
    tmin_padded = tmin - padding
    tmax_padded = tmax + padding
    
    epochs = mne.Epochs(
        raw,
        events=events,
        tmin=tmin_padded,
        tmax=tmax_padded,
        event_id=event_dict,
        baseline=baseline,
        preload=True,
        verbose=False,
        event_repeated="drop",
    )
    
    epochs.crop(tmin=tmin, tmax=tmax)
    
    if motor_channels is not None:
        epochs.pick(motor_channels)
    
    if "CIRCLE ONSET" in event_dict:
        rest_id = event_dict["CIRCLE ONSET"]
        right_id = event_dict.get("ARROW RIGHT ONSET", None)
        left_id = event_dict.get("ARROW LEFT ONSET", None)
        
        y = epochs.events[:, -1].copy()
        y[y == rest_id] = 0
        if right_id is not None:
            y[y == right_id] = 1
        if left_id is not None:
            y[y == left_id] = 2
        epochs.events[:, -1] = y
        
        epochs.event_id = {"rest": 0, "right": 1, "left": 2}
    
    return epochs, event_dict


def reject_bad_epochs(
    epochs: mne.Epochs,
    n_interpolate: List[int] = [1, 2],
    fit: bool = True,
    ar_object: Optional[autoreject.AutoReject] = None,
    save_path: Optional[Path] = None
) -> Tuple[mne.Epochs, Optional[autoreject.AutoReject]]:
    if not AUTOREJECT_AVAILABLE:
        warnings.warn("autoreject not available. Skipping bad epoch rejection.")
        return epochs, None
    
    if ar_object is None:
        ar = autoreject.AutoReject(
            n_interpolate=n_interpolate,
            random_state=RANDOM_STATE,
            n_jobs=1,
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
    
    if save_path is not None and fit:
        import joblib
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(ar, save_path)
        print(f"AutoReject saved to {save_path}")
    
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
    if not AUTOREJECT_AVAILABLE:
        raise ImportError("autoreject not available")
    
    import joblib
    return joblib.load(filepath)


def clean_with_asr(
    epochs: mne.Epochs,
    baseline_data: Optional[mne.io.BaseRaw] = None,
    cutoff: float = 20.0,
    asr_object: Optional = None,
    save_path: Optional[Path] = None
) -> Tuple[mne.Epochs, Optional]:
    if not ASR_AVAILABLE:
        warnings.warn("asrpy not available. Skipping ASR cleaning.")
        return epochs, None
    
    try:
        from asrpy import ASR
        
        if asr_object is None:
            if baseline_data is None:
                print("Warning: No baseline data provided. Using first epoch as baseline.")
                baseline_epoch = epochs[0]
                asr = ASR(sfreq=epochs.info["sfreq"], cutoff=cutoff)
                asr.fit(baseline_epoch)
            else:
                asr = ASR(sfreq=baseline_data.info["sfreq"], cutoff=cutoff)
                asr.fit(baseline_data)
            
            epochs_clean = asr.transform(epochs)
            print(f"ASR: Cleaned {len(epochs)} epochs")
        else:
            asr = asr_object
            epochs_clean = asr.transform(epochs)
            print(f"ASR: Applied to {len(epochs)} epochs")
        
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
    import joblib
    return joblib.load(filepath)


def process_offline(
    raw: mne.io.BaseRaw,
    markers: List[str],
    config: Dict,
    baseline_data: Optional[mne.io.BaseRaw] = None,
    save_ar_path: Optional[Path] = None,
    save_asr_path: Optional[Path] = None
) -> Tuple[mne.Epochs, Optional[autoreject.AutoReject], Optional]:
    print("=" * 60)
    print("OFFLINE PROCESSING PIPELINE")
    print("=" * 60)
    
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


def filter_online(
    window: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float
) -> np.ndarray:
    from scipy import signal
    
    sos = signal.butter(
        4,
        [fmin, fmax],
        btype='bandpass',
        fs=sfreq,
        output='sos'
    )
    
    filtered = signal.sosfilt(sos, window, axis=-1)
    
    return filtered


def create_sliding_window_online(
    data_buffer: np.ndarray,
    window_size: int,
    step: int
) -> Optional[np.ndarray]:
    if data_buffer.shape[1] < window_size:
        return None
    
    window = data_buffer[:, -window_size:]
    return window


def handle_bad_window_online(
    window: np.ndarray,
    max_amplitude: float = 200e-6,
    ar: Optional[autoreject.AutoReject] = None,
    sfreq: float = 250.0
) -> Tuple[np.ndarray, bool]:
    if np.abs(window).max() > max_amplitude:
        return window, True
    
    if ar is not None and AUTOREJECT_AVAILABLE:
        try:
            info = mne.create_info(
                [f"ch{i}" for i in range(window.shape[0])],
                sfreq=sfreq,
                ch_types="eeg"
            )
            window_epoch = window[np.newaxis, :, :]
            epochs = mne.EpochsArray(window_epoch, info, verbose="ERROR")
            
            epochs_clean = ar.transform(epochs)
            
            if len(epochs_clean) == 0:
                return window, True
            
            return epochs_clean.get_data()[0], False
        except Exception:
            return window, True
    
    return window, False


def clean_window_asr_online(
    window: np.ndarray,
    asr_object: Optional = None,
    sfreq: float = 250.0
) -> np.ndarray:
    if not ASR_AVAILABLE or asr_object is None:
        return window
    
    try:
        from asrpy import ASR
        return window
    except Exception:
        return window


def process_window_online(
    window: np.ndarray,
    config: Dict,
    ar: Optional[autoreject.AutoReject] = None,
    asr: Optional = None,
    sfreq: float = 250.0
) -> Tuple[np.ndarray, bool]:
    window_filtered = filter_online(
        window,
        sfreq=sfreq,
        fmin=config["fmin"],
        fmax=config["fmax"]
    )
    
    window_clean, is_bad = handle_bad_window_online(
        window_filtered,
        max_amplitude=config.get("max_amplitude", 200e-6),
        ar=ar,
        sfreq=sfreq
    )
    
    if is_bad:
        return window_clean, True
    
    if config.get("use_asr", False) and asr is not None:
        window_clean = clean_window_asr_online(
            window_clean,
            asr_object=asr,
            sfreq=sfreq
        )
    
    return window_clean, False


def prepare_window_for_model(
    window: np.ndarray
) -> np.ndarray:
    if window.ndim != 2:
        raise ValueError(
            f"Expected 2D window with shape [C, T], got shape {window.shape} "
            f"with {window.ndim} dimensions. "
            "Did you already reshape it? Or did preprocessing fail?"
        )
    
    window_batch = window[np.newaxis, :, :]
    
    return window_batch


__all__ = [
    "filter_raw",
    "create_epochs",
    "reject_bad_epochs",
    "load_autoreject",
    "clean_with_asr",
    "load_asr",
    "process_offline",
    "filter_online",
    "create_sliding_window_online",
    "handle_bad_window_online",
    "clean_window_asr_online",
    "process_window_online",
    "prepare_window_for_model",
]

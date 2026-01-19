"""
Main offline pipeline for MIRepNet model.

This script finetunes and evaluates the MIRepNet foundation model.
- Finetunes on Physionet data
- Tests on separate test data (target subject)
Make sure to set model: "mirepnet" in the config file (bci_config.yaml).

The MIRepNet model:
- Loads pretrained foundation weights
- Finetunes on Physionet training data
- Uses Euclidean Alignment (EA) preprocessing
- Handles channel padding/interpolation automatically
"""

import pickle
import sys
import time
from pathlib import Path

# Add src directory to Python path to allow imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import mne
import numpy as np

# Evaluation
from bci.evaluation.metrics import MetricsTable, compile_metrics

# Data Acquisition
from bci.loading.loading import load_physionet_data, load_target_subject_data
from bci.preprocessing.artefact_removal import ArtefactRemoval

# Preprocessing
from bci.preprocessing.filters import (
    Filter,
)
from bci.preprocessing.windows import epochs_to_windows

# Utils
from bci.utils.bci_config import load_config
from bci.utils.utils import (
    choose_model,
)  # Constructs the model of your choosing (can be easily extended)


def _reject_bad_epochs_with_subject_ids(
    ar: ArtefactRemoval,
    epochs_data: np.ndarray,
    epochs_labels: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ar.rejection_threshold is None:
        raise ValueError("Rejection thresholds have not been computed yet.")
    if epochs_data.shape[0] != epochs_labels.shape[0]:
        raise ValueError("Labels must match number of epochs.")
    if epochs_data.shape[0] != subject_ids.shape[0]:
        raise ValueError("Subject IDs must match number of epochs.")

    good_mask = ~np.any(np.abs(epochs_data) > ar.rejection_threshold, axis=(1, 2))
    rejected = epochs_data.shape[0] - int(good_mask.sum())
    print(
        f"Rejected {rejected} bad epochs out of {epochs_data.shape[0]} total epochs."
    )

    return (
        epochs_data[good_mask],
        epochs_labels[good_mask],
        subject_ids[good_mask],
    )


def _validate_subject_ids(
    epochs_data: np.ndarray,
    epochs_labels: np.ndarray,
    subject_ids: np.ndarray,
    context: str,
) -> None:
    if epochs_data.shape[0] != epochs_labels.shape[0]:
        raise ValueError(f"{context}: labels must match number of epochs.")
    if epochs_data.shape[0] != subject_ids.shape[0]:
        raise ValueError(f"{context}: subject IDs must match number of epochs.")
    if subject_ids.ndim != 1:
        raise ValueError(f"{context}: subject IDs must be a 1D array.")


def _apply_mask_with_subject_ids(
    epochs_data: np.ndarray,
    epochs_labels: np.ndarray,
    subject_ids: np.ndarray,
    mask: np.ndarray,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mask.shape[0] != epochs_data.shape[0]:
        raise ValueError(f"{context}: mask must match number of epochs.")
    _validate_subject_ids(epochs_data, epochs_labels, subject_ids, context)
    return epochs_data[mask], epochs_labels[mask], subject_ids[mask]

if __name__ == "__main__":
    # Load the config file
    current_wd = Path.cwd()  # BCI-Challenge directory

    try:
        config_path = current_wd / "resources" / "configs" / "bci_config.yaml"
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        print("Configuration loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    # Initialize variables
    np.random.seed(config.random_state)
    metrics_table = MetricsTable()
    # MIRepNet model arguments - can be customized here
    model_args = {
        "batch_size": 8,
        "epochs": 10,
        "lr": 0.001,
        "weight_decay": 1e-6,
        "optimizer": "adam",
        "scheduler": "cosine",
        "device": "auto",
        "actual_channels": config.channels if hasattr(config, 'channels') and config.channels else None,
    }  # args for the model chooser (what the model constructor needs)

    # MIRepNet expects 250Hz sampling rate (trained on BNCI2014004 at 250Hz)
    # Override config.fs to 250Hz for filter design
    mirepnet_fs = 250.0
    # Temporarily override config.fs for MIRepNet
    original_fs = config.fs
    config.fs = mirepnet_fs
    filter = Filter(config, online=False)
    # Restore original fs in config (in case it's used elsewhere)
    config.fs = original_fs

    test_data_source_path = (
        current_wd / "data" / config.test
    )  # Path can be defined in config file

    test_data_target_path = (
        current_wd / "data" / "datasets" / config.test
    )  # Path can be defined in config file

    use_test = True  # Whether to test on target subject data
    timings = {}

    # Load training data from Physionet
    x_raw_train, events_train, event_id_train, sub_ids_train = load_physionet_data(
        subjects=config.subjects, root=current_wd, config=config
    )
    print(f"Loaded {len(x_raw_train)} subjects from Physionet for training.")

    # Load target subject data for testing
    x_raw_test, events_test, event_id_test, sub_ids_test = load_target_subject_data(
        root=current_wd,
        source_path=test_data_source_path,
        target_path=test_data_target_path,
        config=config,
        task_type="arrow",
        limit=0,
    )
    print(f"Loaded {len(x_raw_test)} target subject sessions for testing.")

    if len(x_raw_test) == 0:
        print("⚠️ No test data found. Skipping evaluation.")
        use_test = False

    # Training: Filter the data and create epochs
    # Note: MIRepNet model expects 250Hz sampling rate (trained on BNCI2014004 at 250Hz)
    # We need to resample both Physionet (160Hz) and target subject data to 250Hz
    target_sfreq = 250.0
    all_train_windows = []
    all_train_labels = []
    all_train_subject_ids = []
    for raw, events, sub_id in zip(x_raw_train, events_train, sub_ids_train):
        # Resample Physionet data to 250Hz for MIRepNet
        if raw.info['sfreq'] != target_sfreq:
            print(f"Resampling Physionet data from {raw.info['sfreq']}Hz to {target_sfreq}Hz for MIRepNet")
            raw.resample(target_sfreq)
        
        # Filter the data
        filtered_raw = filter.apply_filter_offline(raw)

        # Create the epochs
        epochs = mne.Epochs(
            filtered_raw,
            events,
            event_id=event_id_train,
            tmin=0.5,
            tmax=4.0,
            preload=True,
            baseline=None,
        )

        # Extract windowed epochs per subject to keep subject IDs aligned
        X_subject_windows, y_subject_windows = epochs_to_windows(
            epochs,
            window_size=config.window_size,
            step_size=config.step_size,
        )
        if X_subject_windows.shape[0] == 0:
            continue

        all_train_windows.append(X_subject_windows)
        all_train_labels.append(y_subject_windows)
        all_train_subject_ids.append(
            np.full(X_subject_windows.shape[0], sub_id, dtype=np.int64)
        )

    if len(all_train_windows) == 0:
        raise RuntimeError("No training windows extracted after preprocessing.")

    X_train_windows = np.concatenate(all_train_windows, axis=0)
    y_train_windows = np.concatenate(all_train_labels, axis=0)
    subject_ids_train = np.concatenate(all_train_subject_ids, axis=0)
    _validate_subject_ids(
        X_train_windows, y_train_windows, subject_ids_train, "train windows"
    )

    ar = ArtefactRemoval()
    ar.get_rejection_thresholds(X_train_windows, config)
    X_train_clean, y_train_clean, subject_ids_train_clean = (
        _reject_bad_epochs_with_subject_ids(
            ar, X_train_windows, y_train_windows, subject_ids_train
        )
    )

    # Get unique labels from training data (used as reference for label mapping)
    train_labels_unique = np.unique(y_train_clean)
    train_labels_sorted = np.sort(train_labels_unique)
    
    # Create label mapping: original label -> remapped label (0-indexed consecutive)
    # This ensures consistent labeling even if original labels are not 0-indexed
    label_mapping = {old_label: idx for idx, old_label in enumerate(train_labels_sorted)}
    reverse_label_mapping = {idx: old_label for old_label, idx in label_mapping.items()}
    
    print(f"\nLabel remapping:")
    print(f"  Original training labels: {sorted(train_labels_sorted.tolist())}")
    print(f"  Remapping to: {list(range(len(train_labels_sorted)))}")
    print(f"  Mapping: {label_mapping}")
    
    # Remap training labels to 0-indexed consecutive integers
    y_train_remapped = np.array([label_mapping[label] for label in y_train_clean], dtype=np.int64)
    
    print(f"  Training labels after remapping: {sorted(np.unique(y_train_remapped).tolist())}")

    # Construct Final Model
    clf = choose_model(config.model, model_args)

    # Train the final model on all clean training data (using remapped labels)
    print("\nTraining final model on all Physionet training data...")
    start_train_time = time.time() * 1000
    clf.fit(X_train_clean, y_train_remapped, subject_ids=subject_ids_train_clean)
    end_train_time = time.time() * 1000

    # Test on the holdout test set
    if use_test:
        # Prepare test data - Filter, Epoch and Window, Remove Artifacts
        # Note: MIRepNet expects 250Hz, so resample test data to match
        target_sfreq = 250.0
        all_test_windows = []
        all_test_labels = []
        all_test_subject_ids = []
        start_total_time = time.time() * 1000
        for raw, events, sub_id in zip(x_raw_test, events_test, sub_ids_test):
            # Resample test data to 250Hz for MIRepNet
            # Note: load_target_subject_data resamples to 160Hz, but MIRepNet needs 250Hz
            if raw.info['sfreq'] != target_sfreq:
                print(f"Resampling test data from {raw.info['sfreq']}Hz to {target_sfreq}Hz for MIRepNet")
                raw.resample(target_sfreq)
            
            # Filter the data
            filtered_raw = filter.apply_filter_offline(raw)

            # Create the epochs for testing
            epochs = mne.Epochs(
                filtered_raw,
                events,
                event_id=event_id_test,
                tmin=0.5,
                tmax=4.0,
                preload=True,
                baseline=None,
            )

            # Extract windowed epochs per subject to keep subject IDs aligned
            X_subject_windows, y_subject_windows = epochs_to_windows(
                epochs,
                window_size=config.window_size,
                step_size=config.step_size,
            )
            if X_subject_windows.shape[0] == 0:
                continue

            all_test_windows.append(X_subject_windows)
            all_test_labels.append(y_subject_windows)
            all_test_subject_ids.append(
                np.full(X_subject_windows.shape[0], sub_id, dtype=np.int64)
            )

        if len(all_test_windows) == 0:
            raise RuntimeError("No test windows extracted after preprocessing.")

        X_test_windows = np.concatenate(all_test_windows, axis=0)
        y_test_windows = np.concatenate(all_test_labels, axis=0)
        subject_ids_test = np.concatenate(all_test_subject_ids, axis=0)
        _validate_subject_ids(
            X_test_windows, y_test_windows, subject_ids_test, "test windows"
        )

        # Also clean test data using the same thresholds
        X_test_clean, y_test_clean, subject_ids_test_clean = (
            _reject_bad_epochs_with_subject_ids(
                ar, X_test_windows, y_test_windows, subject_ids_test
            )
        )

        # Remap test labels using the same mapping as training
        test_labels_unique = np.unique(y_test_clean)
        print(f"\nTest labels before remapping: {sorted(test_labels_unique.tolist())}")
        
        # Check if test has any labels not in training
        test_labels_not_in_training = set(test_labels_unique) - set(train_labels_sorted)
        if test_labels_not_in_training:
            print(f"⚠️  WARNING: Test data contains labels not in training: {sorted(test_labels_not_in_training)}")
            print(f"   These labels will be mapped to closest training label or removed.")
            # For now, we'll skip samples with unknown labels
            valid_mask = np.isin(y_test_clean, train_labels_sorted)
            if not np.all(valid_mask):
                print(f"   Removing {np.sum(~valid_mask)} samples with unknown labels from test set.")
                (
                    X_test_clean,
                    y_test_clean,
                    subject_ids_test_clean,
                ) = _apply_mask_with_subject_ids(
                    X_test_clean,
                    y_test_clean,
                    subject_ids_test_clean,
                    valid_mask,
                    "test label filtering",
                )
        
        # Remap test labels using the same mapping
        y_test_remapped = np.array([label_mapping[label] for label in y_test_clean], dtype=np.int64)
        print(f"  Test labels after remapping: {sorted(np.unique(y_test_remapped).tolist())}")

        # Evaluate on holdout test set (using remapped labels)
        print("\nEvaluating on holdout test set...")
        start_eval_time = time.time() * 1000
        test_predictions = clf.predict(
            X_test_clean, subject_ids=subject_ids_test_clean
        )
        end_eval_time = time.time() * 1000
        test_probabilities = clf.predict_proba(
            X_test_clean, subject_ids=subject_ids_test_clean
        )

        end_total_time = time.time() * 1000

        # Remap predictions back to original label space for metrics/comparison
        # (though metrics should work fine with remapped labels)
        test_predictions_original = np.array([reverse_label_mapping[pred] for pred in test_predictions])

        # Compute metrics for test set (using remapped labels for consistency)

        test_metrics = compile_metrics(
            y_true=y_test_remapped,
            y_pred=test_predictions,
            y_prob=test_probabilities,
            timings={
                "train_time": end_train_time - start_train_time,
                "infer_latency": (end_eval_time - start_eval_time)
                / max(1, len(y_test_clean)),
                "total_latency": (end_total_time - start_total_time)
                / max(1, len(y_test_clean)),
                "filter_latency": filter.get_filter_latency(),
            },
            n_classes=len(train_labels_sorted),
        )

        # Add dataset label after computing metrics
        test_metrics["Dataset"] = "Test (Holdout)"

        metrics_table.add_rows([test_metrics])
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        metrics_table.display()

    # Save the Model and other Objects
    # MIRepNet uses .pt extension for PyTorch models
    model_path = Path.cwd() / "resources" / "models" / "mirepnet_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(model_path))
    print(f"Model saved to: {model_path}")

    # Save the Artefact Removal Object
    ar_path = Path.cwd() / "resources" / "models" / "artefact_removal.pkl"
    with open(ar_path, "wb") as f:
        pickle.dump(ar, f)
    print(f"Artefact Removal object saved to: {ar_path}")

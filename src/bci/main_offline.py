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
import pandas as pd
from sklearn.model_selection import GroupKFold

# Evaluation
from bci.evaluation.metrics import MetricsTable, compile_metrics

# Data Acquisition
from bci.loading.loading import load_physionet_data, load_target_subject_data
from bci.preprocessing.artefact_removal import ArtefactRemoval

# Preprocessing
from bci.preprocessing.filters import (
    Filter,
)
from bci.preprocessing.windows import epochs_to_windows, epochs_windows_from_fold

# Utils
from bci.utils.bci_config import load_config
from bci.utils.utils import (
    choose_model,
)  # Constructs the model of your choosing (can be easily extended)

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
    model_args = {
        "cov_est": "lwf"
    }  # args for the model chooser (what the model constructor needs)
    gkf = None
    if config.n_folds < 2:
        print("No cross-validation will be performed.")
    else:
        gkf = GroupKFold(n_splits=config.n_folds)  # Cross-Validation splitter

    filter = Filter(config, online=False)

    test_data_source_path = (
        current_wd / "data" / config.test
    )  # Path can be defined in config file

    test_data_target_path = (
        current_wd / "data" / "datasets" / config.test
    )  # Path can be defined in config file

    use_test = True  # Whether to test on target subject data after CV
    timings = {}

    # Load training data
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
        task_type="",
        limit=0,
    )
    print(f"Loaded {len(x_raw_test)} target subject sessions for testing.")

    # Training: Filter the data and create epochs with metadata for grouped CV
    all_epochs_list = []
    for raw, events, sub_id in zip(x_raw_train, events_train, sub_ids_train):
        # Filter the data
        filtered_raw = filter.apply_filter_offline(raw)

        # Create the epochs for CV with metadata
        epochs = mne.Epochs(
            filtered_raw,
            events,
            event_id=event_id_train,
            tmin=0.5,
            tmax=4.0,
            preload=True,
            baseline=None,
        )

        # Attach metadata
        metadata = pd.DataFrame(
            {
                "subject_id": [sub_id] * len(epochs),
                "condition": epochs.events[:, 2],
            }
        )
        epochs.metadata = metadata

        all_epochs_list.append(epochs)

    # Prepare Epochs with Metadata for Grouped CV
    combined_epochs = mne.concatenate_epochs(all_epochs_list)
    X_train = combined_epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y_train = combined_epochs.events[:, 2]  # The labels (e.g., 1, 2, 3)
    groups = (
        combined_epochs.metadata["subject_id"].values
        if combined_epochs.metadata is not None
        else None
    )

    # Grouped K-Fold Cross-Validation
    if config.n_folds >= 2 and gkf is not None and groups is not None:
        cv_metrics_list = []  # To compute the mean metrics over folds
        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(X_train, y_train, groups=groups)
        ):
            # Epoch the data into windows
            fold_windowed_epochs = epochs_windows_from_fold(
                combined_epochs,
                train_idx,
                val_idx,
                window_size=config.window_size,
                step_size=config.step_size,
            )

            X_train_fold, y_train_fold = (
                fold_windowed_epochs["X_train"],
                fold_windowed_epochs["y_train"],
            )
            X_val_fold, y_val_fold = (
                fold_windowed_epochs["X_val"],
                fold_windowed_epochs["y_val"],
            )

            # Remove artifacts within each fold
            ar = ArtefactRemoval()
            ar.get_rejection_thresholds(X_train_fold, config)

            X_train_clean, y_train_clean = ar.reject_bad_epochs(
                X_train_fold, y_train_fold
            )
            X_val_clean, y_val_clean = ar.reject_bad_epochs(X_val_fold, y_val_fold)

            # Train and evaluate the model within each fold
            clf = choose_model(config.model, model_args)
            clf.fit(X_train_clean, y_train_clean)

            # Evaluate on validation fold
            fold_predictions = clf.predict(X_val_clean)
            fold_probabilities = clf.predict_proba(X_val_clean)

            # Compute metrics for this fold
            fold_metrics = compile_metrics(
                y_true=y_val_clean,
                y_pred=fold_predictions,
                y_prob=fold_probabilities,
                timings=None,
                n_classes=len(event_id_train),
            )

            cv_metrics_list.append(fold_metrics)
            print(f"Fold {fold_idx} Accuracy: {fold_metrics['Acc.']:.4f}")

        # CV Results
        # TODO: Discuss with team what metrics are relevant from CV. For now just the performance ones are computed

        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS (Mean ± Std)")
        print("=" * 60)

        cv_mean_metrics = {}
        cv_std_metrics = {}
        metric_keys = cv_metrics_list[0].keys()

        for key in metric_keys:
            values = [m[key] for m in cv_metrics_list]
            if key in [
                "Train Time (ms)",
                "Infer. Time (ms)",
                "Avg. Filter Latency (ms)",
                "ITR (bits/min)",
            ]:
                cv_mean_metrics[key] = round(np.mean(values), 2)
                cv_std_metrics[key] = round(np.std(values), 2)
            else:
                cv_mean_metrics[key] = round(np.mean(values), 4)
                cv_std_metrics[key] = round(np.std(values), 4)

        # Create CV summary row
        cv_summary = {}
        cv_summary["Dataset"] = "CV (Training)"
        for key in metric_keys:
            if key in [
                "Train Time (ms)",
                "Infer. Time (ms)",
                "Avg. Filter Latency (ms)",
                "ITR (bits/min)",
            ]:
                cv_summary[key] = (
                    f"{cv_mean_metrics[key]:.2f} ± {cv_std_metrics[key]:.2f}"
                )
            else:
                cv_summary[key] = (
                    f"{cv_mean_metrics[key]:.4f} ± {cv_std_metrics[key]:.4f}"
                )

        # Add CV results to metrics table
        metrics_table.add_rows([cv_summary])
        metrics_table.display()

    # Results on Holdout

    # Extract the windowed epochs
    X_train_windows, y_train_windows = epochs_to_windows(
        combined_epochs,
        window_size=config.window_size,
        step_size=config.step_size,
    )

    ar = ArtefactRemoval()
    ar.get_rejection_thresholds(X_train_windows, config)
    X_train_clean, y_train_clean = ar.reject_bad_epochs(
        X_train_windows, y_train_windows
    )

    # Construct Final Model
    clf = choose_model(config.model, model_args)

    # Train the final model on all clean training data
    print("\nTraining final model on all training data...")
    start_train_time = time.time() * 1000
    clf.fit(X_train_clean, y_train_clean)
    end_train_time = time.time() * 1000

    # Test on the holdout test set
    if use_test:
        # Prepare test data - Filter, Epoch and Window, Remove Artifacts
        all_test_epochs_list = []
        start_total_time = time.time() * 1000
        for raw, events, sub_id in zip(x_raw_test, events_test, sub_ids_test):
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

            # TODO: Should we attach metadata?

            all_test_epochs_list.append(epochs)

        combined_test_epochs = mne.concatenate_epochs(all_test_epochs_list)
        test_epochs = combined_test_epochs.get_data()
        test_labels = combined_test_epochs.events[:, 2]

        # Extract windowed epochs for testing
        X_test_windows, y_test_windows = epochs_to_windows(
            combined_test_epochs,
            window_size=config.window_size,
            step_size=config.step_size,
        )

        # Also clean test data using the same thresholds
        X_test_clean, y_test_clean = ar.reject_bad_epochs(
            X_test_windows, y_test_windows
        )

        # Evaluate on holdout test set
        print("Evaluating on holdout test set...")
        start_eval_time = time.time() * 1000
        test_predictions = clf.predict(X_test_clean)
        end_eval_time = time.time() * 1000
        test_probabilities = clf.predict_proba(X_test_clean)

        end_total_time = time.time() * 1000

        # Compute metrics for test set

        test_metrics = compile_metrics(
            y_true=y_test_clean,
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
            n_classes=len(event_id_train),
        )

        # Add dataset label after computing metrics
        test_metrics["Dataset"] = "Test (Holdout)"

        metrics_table.add_rows([test_metrics])
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        metrics_table.display()

    # Save the Model and other Objects
    model_path = Path.cwd() / "resources" / "models" / "model.pkl"
    clf.save(model_path)
    print(f"Model saved to: {model_path}")

    # Save the Artefact Removal Object
    ar_path = Path.cwd() / "resources" / "models" / "artefact_removal.pkl"
    with open(ar_path, "wb") as f:
        pickle.dump(ar, f)
    print(f"Artefact Removal object saved to: {ar_path}")

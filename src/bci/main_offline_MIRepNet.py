"""
Main offline pipeline for MIRepNet model with 5-fold cross-validation.

This script finetunes and evaluates the MIRepNet foundation model on target subject data.
Uses 5-fold cross-validation: for each fold, 4 folds are used for finetuning and 1 fold for testing.
Make sure to set model: "mirepnet" in the config file (bci_config.yaml).

The MIRepNet model:
- Loads pretrained foundation weights
- Finetunes on 80% of the data (4 folds)
- Tests on the remaining 20% (1 fold)
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
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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

    filter = Filter(config, online=False)

    test_data_path = (
        current_wd / "data" / config.test
    )  # Path can be defined in config file

    # Load target subject data (will be used for 5-fold CV)
    x_raw_test, events_test, event_id_test, sub_ids_test = load_target_subject_data(
        root=current_wd,
        source_path=test_data_path,
        config=config,
        task_type="arrow",
        limit=0,
    )
    print(f"Loaded {len(x_raw_test)} target subject sessions.")
    print("Note: Using 5-fold cross-validation on target subject data.")

    # Process test data: Filter, Epoch and Window
    print("\nProcessing test data...")
    all_test_epochs_list = []
    for raw, events, sub_id in zip(x_raw_test, events_test, sub_ids_test):
        # Filter the data
        filtered_raw = filter.apply_filter_offline(raw)

        # Create the epochs
        epochs = mne.Epochs(
            filtered_raw,
            events,
            event_id=event_id_test,
            tmin=0.5,
            tmax=4.0,
            preload=True,
            baseline=None,
        )

        # Attach metadata for potential grouping
        metadata = pd.DataFrame(
            {
                "subject_id": [sub_id] * len(epochs),
                "condition": epochs.events[:, 2],
            }
        )
        epochs.metadata = metadata

        all_test_epochs_list.append(epochs)

    # Combine all test epochs
    combined_test_epochs = mne.concatenate_epochs(all_test_epochs_list)
    
    # Extract windowed epochs
    X_test_windows, y_test_windows = epochs_to_windows(
        combined_test_epochs,
        window_size=config.window_size,
        step_size=config.step_size,
    )
    
    print(f"Total windows from test data: {len(X_test_windows)}")
    
    # 5-Fold Cross-Validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_state)
    cv_metrics_list = []
    
    print(f"\nStarting {n_folds}-fold cross-validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_test_windows, y_test_windows)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Split data for this fold
        X_finetune_fold = X_test_windows[train_idx]
        y_finetune_fold = y_test_windows[train_idx]
        X_test_fold = X_test_windows[val_idx]
        y_test_fold = y_test_windows[val_idx]
        
        print(f"Finetuning set: {len(X_finetune_fold)} windows")
        print(f"Test set: {len(X_test_fold)} windows")
        
        # Remove artifacts from finetuning data
        ar = ArtefactRemoval()
        ar.get_rejection_thresholds(X_finetune_fold, config)
        X_finetune_clean, y_finetune_clean = ar.reject_bad_epochs(
            X_finetune_fold, y_finetune_fold
        )
        
        # Also clean test data using the same thresholds
        X_test_clean, y_test_clean = ar.reject_bad_epochs(X_test_fold, y_test_fold)
        
        print(f"After artifact removal - Finetuning: {len(X_finetune_clean)}, Test: {len(X_test_clean)}")

        # Construct and train model on finetuning data
        clf = choose_model(config.model, model_args)

        # Finetune the model on finetuning fold
        print(f"\nFinetuning model on fold {fold_idx + 1} training data...")
        start_train_time = time.time() * 1000
        clf.fit(X_finetune_clean, y_finetune_clean)
        end_train_time = time.time() * 1000

        # Evaluate on the test fold
        print(f"Evaluating on fold {fold_idx + 1} test data...")
        start_eval_time = time.time() * 1000
        test_predictions = clf.predict(X_test_clean)
        end_eval_time = time.time() * 1000
        test_probabilities = clf.predict_proba(X_test_clean)
        end_total_time = time.time() * 1000

        # Compute metrics for this fold
        fold_metrics = compile_metrics(
            y_true=y_test_clean,
            y_pred=test_predictions,
            y_prob=test_probabilities,
            timings={
                "train_time": end_train_time - start_train_time,
                "infer_latency": (end_eval_time - start_eval_time)
                / max(1, len(y_test_clean)),
                "total_latency": (end_total_time - start_eval_time)
                / max(1, len(y_test_clean)),
                "filter_latency": filter.get_filter_latency(),
            },
            n_classes=len(event_id_test),
        )
        
        fold_metrics["Dataset"] = f"Fold {fold_idx + 1}"
        cv_metrics_list.append(fold_metrics)
        print(f"Fold {fold_idx + 1} Accuracy: {fold_metrics['Acc.']:.4f}")
    
    # Compute CV summary statistics
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS (Mean ± Std)")
    print("=" * 60)
    
    cv_mean_metrics = {}
    cv_std_metrics = {}
    metric_keys = cv_metrics_list[0].keys()
    
    for key in metric_keys:
        if key == "Dataset":
            continue
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
    cv_summary["Dataset"] = "CV (Mean ± Std)"
    for key in metric_keys:
        if key == "Dataset":
            continue
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
    
    # Add all fold results and CV summary to metrics table
    metrics_table.add_rows(cv_metrics_list)
    metrics_table.add_rows([cv_summary])
    metrics_table.display()
    
    # Save the last fold's model (or you could save the best fold's model)
    print("\n" + "=" * 60)
    print("Saving final model (from last fold)...")
    print("=" * 60)

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

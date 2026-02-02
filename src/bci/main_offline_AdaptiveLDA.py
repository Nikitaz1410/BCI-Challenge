"""
Offline Training and Evaluation Script for Hybrid 2-Stage LDA.

1. Loads training/test data using create_subject_train_set/create_subject_test_set
2. Trains a HybridLDA classifier (2-stage: Rest vs Movement, Left vs Right)
3. Performs cross-validation with file-wise splits
4. Simulates offline adaptation to compare accuracy with and without adaptation
5. Saves the trained model to hybrid_lda.pkl

"""

import pickle
import sys
import time
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

# Evaluation
from bci.Evaluation.metrics import MetricsTable, compile_metrics

# Data Acquisition
from bci.loading.loading import (
    load_physionet_data,
    load_target_subject_data,
    create_subject_train_set,
    create_subject_test_set
)

# Preprocessing
from bci.Preprocessing.filters import Filter
from bci.Preprocessing.windows import epochs_to_windows, epochs_windows_from_fold
from bci.Preprocessing.artefact_removal import ArtefactRemoval

# Models - HybridLDA for 2-stage classification
from bci.Models.AdaptiveLDA_modules.hybrid_lda import HybridLDA
from bci.Models.AdaptiveLDA_modules.feature_extraction import extract_log_bandpower_features

# Utils
from bci.utils.bci_config import load_config


# =============================================================================
# Feature Extraction Helper (shared between offline and online)
# =============================================================================
def extract_features(signals, sfreq):
    """
    Shared feature extraction function.

    Uses log-bandpower features (mu: 8-12 Hz, beta: 13-30 Hz).

    Parameters:
    -----------
    signals : np.ndarray
        Shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz

    Returns:
    --------
    features : np.ndarray
        Shape (n_trials, n_features) or (n_features,)
    """
    return extract_log_bandpower_features(signals, sfreq=sfreq, mu_band=(8, 12), beta_band=(13, 30))


# =============================================================================
# Offline Adaptation Simulation
# =============================================================================
def simulate_offline_adaptation(hybrid_lda, X_test, y_test, sfreq):
    """
    Simulate online adaptation in an offline setting.

    Processes test samples one by one:
    1. Predict (before adaptation)
    2. Update model with true label (after seeing the label)
    3. Compare accuracy with and without adaptation

    Parameters:
    -----------
    hybrid_lda : HybridLDA
        The trained classifier (will be modified in place)
    X_test : np.ndarray
        Test signals, shape (n_samples, n_channels, n_times)
    y_test : np.ndarray
        True labels, shape (n_samples,)
    sfreq : float
        Sampling frequency

    Returns:
    --------
    results : dict
        Contains predictions, accuracies, and adaptation statistics
    """
    n_samples = len(X_test)
    predictions_no_adapt = np.zeros(n_samples, dtype=int)
    predictions_with_adapt = np.zeros(n_samples, dtype=int)

    print(f"\n  Simulating adaptation on {n_samples} test samples...")

    for i in range(n_samples):
        # Extract features for this sample
        features = extract_features(X_test[i:i+1], sfreq)  # Shape: (1, n_features)

        # Predict BEFORE adaptation (for comparison)
        pred = hybrid_lda.predict(features)[0]
        predictions_no_adapt[i] = pred
        predictions_with_adapt[i] = pred  # Same prediction initially

        # Update model with true label (simulates receiving feedback)
        true_label = int(y_test[i])
        hybrid_lda.update(true_label, features[0])

        # Progress update
        if (i + 1) % 100 == 0:
            acc_so_far = (predictions_with_adapt[:i+1] == y_test[:i+1]).mean()
            print(f"    Processed {i+1}/{n_samples} samples, running acc: {acc_so_far:.3f}")

    # Now re-predict with the adapted model for comparison
    # (In real online use, adaptation happens AFTER prediction, so this shows
    #  what would happen if we processed the same data with the adapted model)
    features_all = extract_features(X_test, sfreq)
    predictions_post_adapt = hybrid_lda.predict(features_all)

    acc_no_adapt = (predictions_no_adapt == y_test).mean()
    acc_with_adapt = (predictions_with_adapt == y_test).mean()
    acc_post_adapt = (predictions_post_adapt == y_test).mean()

    stats = hybrid_lda.get_update_stats()

    return {
        'predictions_no_adapt': predictions_no_adapt,
        'predictions_with_adapt': predictions_with_adapt,
        'predictions_post_adapt': predictions_post_adapt,
        'acc_no_adapt': acc_no_adapt,
        'acc_with_adapt': acc_with_adapt,
        'acc_post_adapt': acc_post_adapt,
        'update_stats': stats
    }


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Load the config file
    # Detect project root: handle both workspace root and BCI-Challenge subdirectory
    # Script is at: [workspace]/BCI-Challenge/src/bci/main_offline_AdaptiveLDA.py
    script_dir = Path(__file__).parent.parent.parent  # Goes up 3 levels from script
    
    # Check if we're in a BCI-Challenge subdirectory (workspace structure)
    # The script is at: BCI-Challenge/src/bci/main_offline_AdaptiveLDA.py
    # So script_dir should be BCI-Challenge directory
    # Check if script_dir contains "src" and "data" directories to confirm it's the project root
    if (script_dir / "src").exists() and (script_dir / "data").exists():
        # This is the BCI-Challenge project root
        current_wd = script_dir
    elif (script_dir / "BCI-Challenge" / "src").exists() and (script_dir / "BCI-Challenge" / "data").exists():
        # We're in workspace root, need to go into BCI-Challenge
        current_wd = script_dir / "BCI-Challenge"
    else:
        # Fallback: assume script_dir is correct
        current_wd = script_dir

    try:
        config_path = current_wd / "resources" / "configs" / "bci_config.yaml"
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        print("Configuration loaded successfully!")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Initialize variables
    np.random.seed(config.random_state)
    metrics_table = MetricsTable()

    gkf = None
    if config.n_folds < 2:
        print("No cross-validation will be performed.")
    else:
        gkf = GroupKFold(n_splits=config.n_folds)

    filter_obj = Filter(config, online=False)

    # Paths for target subject data (matching friend's approach)
    test_data_source_path = current_wd / "data" / "eeg" / config.target
    test_data_target_path = current_wd / "data" / "datasets" / config.target

    use_test = True
    use_adaptation_simulation = True  # Whether to simulate offline adaptation

    # ==========================================================================
    # 1. Load All Target Subject Data 
    # ==========================================================================
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    all_target_raws, all_target_events, target_event_id, target_sub_ids, target_metadata = load_target_subject_data(
        root=current_wd,
        source_path=test_data_source_path,
        target_path=test_data_target_path,
        resample=None,
    )
    print(f"Loaded {len(all_target_raws)} sessions from target subject data.")

    # ==========================================================================
    # 2. Create Training Set 
    # ==========================================================================
    x_raw_train, events_train, train_filenames, sub_ids_train, train_indices = create_subject_train_set(
        config,
        all_target_raws,
        all_target_events,
        target_metadata["filenames"],
        num_general=3,
        num_dino=5,
        num_supression=0,
        shuffle=True
    )
    print(f"Created training set with {len(x_raw_train)} files.")

    # ==========================================================================
    # 3. Create Test Set 
    # ==========================================================================
    x_raw_test, events_test, test_filenames, sub_ids_test = create_subject_test_set(
        config,
        all_target_raws,
        all_target_events,
        target_metadata["filenames"],
        exclude_indices=train_indices,
        num_general=0,
        num_dino=4,
        num_supression=0,
        shuffle=False
    )
    print(f"Created test set with {len(x_raw_test)} files.")

    # ==========================================================================
    # 4. Preprocess Training Data: Filter, Epoch, Add Metadata
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PREPROCESSING TRAINING DATA")
    print("=" * 60)

    all_epochs_list = []
    for raw, events, sub_id, filename in zip(x_raw_train, events_train, sub_ids_train, train_filenames):
        # FILTERING: Filter the data by using the mne apply function method
        filtered_raw = raw.copy()
        filtered_raw.apply_function(filter_obj.apply_filter_offline)

        # CHANNEL REMOVAL: Remove unnecessary channels like noise sources
        filtered_raw.drop_channels(config.remove_channels)

        # EPOCHING: Create the epochs for CV with metadata
        epochs = mne.Epochs(
            filtered_raw,
            events,
            event_id=target_event_id,
            tmin=0.3,  # Start at 0.3 to avoid VEP/ERP due to the Visual Cues
            tmax=4.0,
            preload=True,
            baseline=None,
        )

        # Attach metadata for file-wise CV
        metadata = pd.DataFrame({
            "subject_id": [sub_id] * len(epochs),
            "filename": [filename] * len(epochs),
            "condition": epochs.events[:, 2],
        })
        epochs.metadata = metadata
        all_epochs_list.append(epochs)

    # Combine all epochs
    combined_epochs = mne.concatenate_epochs(all_epochs_list)
    
    X_train = combined_epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y_train = combined_epochs.events[:, 2]  # Labels (0, 1, 2) from target_event_id
    groups = combined_epochs.metadata["filename"].values if combined_epochs.metadata is not None else None

    # Labels are already [0, 1, 2] from target_event_id, no mapping needed
    print(f"Training data: {X_train.shape[0]} epochs, {X_train.shape[1]} channels, labels: {np.unique(y_train)}")

    # ==========================================================================
    # 5. Cross-Validation with HybridLDA
    # ==========================================================================
    if config.n_folds >= 2 and gkf is not None and groups is not None:
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION WITH HYBRID LDA")
        print("=" * 60)

        cv_metrics_list = []
        cv_confusion_matrices = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
            print(f"\n--- Fold {fold_idx + 1}/{config.n_folds} ---")

            # Window the data for this fold
            fold_windowed = epochs_windows_from_fold(
                combined_epochs,
                groups,
                train_idx,
                val_idx,
                window_size=config.window_size,
                step_size=config.step_size,
            )

            X_train_fold = fold_windowed["X_train"]
            y_train_fold = fold_windowed["y_train"]  # Labels already [0, 1, 2]
            X_val_fold = fold_windowed["X_val"]
            y_val_fold = fold_windowed["y_val"]  # Labels already [0, 1, 2]

            # Extract features
            train_features = extract_features(X_train_fold, config.fs)
            val_features = extract_features(X_val_fold, config.fs)

            # Train HybridLDA
            fold_clf = HybridLDA(
                move_threshold=0.5,
                reg=1e-2,
                shrinkage_alpha=0.1,
                uc_mu=0.4 * 2**-6,
                use_improved_composition=True
            )

            start_train = time.time() * 1000
            fold_clf.fit(train_features, y_train_fold)
            end_train = time.time() * 1000

            # Predict on validation fold
            start_eval = time.time() * 1000
            fold_preds = fold_clf.predict(val_features)
            end_eval = time.time() * 1000
            fold_probs = fold_clf.predict_proba(val_features)

            # Compute metrics
            fold_metrics = compile_metrics(
                y_true=y_val_fold,
                y_pred=fold_preds,
                y_prob=fold_probs,
                timings={
                    "train_time": end_train - start_train,
                    "infer_latency": (end_eval - start_eval) / max(1, len(y_val_fold)),
                    "total_latency": (end_eval - start_eval) / max(1, len(y_val_fold)),
                    "filter_latency": filter_obj.get_filter_latency(),
                },
                n_classes=3,
            )

            cv_metrics_list.append(fold_metrics)
            cv_confusion_matrices.append(confusion_matrix(y_val_fold, fold_preds))
            print(f"Fold {fold_idx + 1} Accuracy: {fold_metrics['Acc.']:.4f}")

        # CV Summary
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS (Mean +/- Std)")
        print("=" * 60)

        cv_mean_metrics = {}
        cv_std_metrics = {}
        metric_keys = cv_metrics_list[0].keys()

        for key in metric_keys:
            values = [m[key] for m in cv_metrics_list]
            cv_mean_metrics[key] = np.mean(values)
            cv_std_metrics[key] = np.std(values)

        cv_summary = {"Dataset": "CV (Training)"}
        for key in metric_keys:
            if "Time" in key or "Latency" in key or "ITR" in key:
                cv_summary[key] = f"{cv_mean_metrics[key]:.2f} +/- {cv_std_metrics[key]:.2f}"
            else:
                cv_summary[key] = f"{cv_mean_metrics[key]:.4f} +/- {cv_std_metrics[key]:.4f}"

        metrics_table.add_rows([cv_summary])
        metrics_table.display()

        # Save average confusion matrix
        avg_cm = np.mean(cv_confusion_matrices, axis=0).astype(int)
        plt.figure(figsize=(8, 6))
        sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Rest', 'Left', 'Right'],
                    yticklabels=['Rest', 'Left', 'Right'])
        plt.xlabel('Predicted', fontsize=12, fontweight='bold')
        plt.ylabel('True', fontsize=12, fontweight='bold')
        plt.title('HybridLDA CV - Average Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        cm_path = current_wd / "hybrid_lda_cv_confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        print(f"\nCV confusion matrix saved: {cm_path}")
        plt.close()

    # ==========================================================================
    # 6. Train Final HybridLDA on All Training Data
    # ==========================================================================
    print("\n" + "=" * 60)
    print("TRAINING FINAL HYBRID LDA MODEL")
    print("=" * 60)

    # Window all training data
    X_train_windows, y_train_windows, _ = epochs_to_windows(
        combined_epochs,
        groups,
        window_size=config.window_size,
        step_size=config.step_size,
    )
    # Labels are already [0, 1, 2] from epochs_to_windows, no mapping needed

    # Extract features
    train_features_all = extract_features(X_train_windows, config.fs)

    # Train final model
    clf = HybridLDA(
        move_threshold=0.5,
        reg=1e-2,
        shrinkage_alpha=0.1,
        uc_mu=0.4 * 2**-6,
        use_improved_composition=True
    )

    start_train = time.time() * 1000
    clf.fit(train_features_all, y_train_windows)
    end_train = time.time() * 1000

    print(f"Training completed in {end_train - start_train:.2f} ms")
    print(f"Stage info: {clf.get_stage_info()}")

    # ==========================================================================
    # 7. Evaluate on Holdout Test Set
    # ==========================================================================
    if use_test and len(x_raw_test) > 0:
        print("\n" + "=" * 60)
        print("EVALUATION ON HOLDOUT TEST SET")
        print("=" * 60)

        # Preprocess test data
        all_test_epochs_list = []
        for raw, events, sub_id in zip(x_raw_test, events_test, sub_ids_test):
            # FILTERING: Filter the data by using the mne apply function method
            filtered_raw = raw.copy()
            filtered_raw.apply_function(filter_obj.apply_filter_offline)

            # CHANNEL REMOVAL: Remove unnecessary channels like noise sources
            filtered_raw.drop_channels(config.remove_channels)

            # Create the epochs for testing
            epochs = mne.Epochs(
                filtered_raw,
                events,
                event_id=target_event_id,
                tmin=0.3,  # Start at 0.3 to avoid VEP/ERP due to the Visual Cues (match training)
                tmax=4.0,
                preload=True,
                baseline=None,
            )
            all_test_epochs_list.append(epochs)

        combined_test_epochs = mne.concatenate_epochs(all_test_epochs_list)
        test_labels = combined_test_epochs.events[:, 2]

        # Window test data
        # Create dummy groups for test data (one per epoch)
        test_groups = np.arange(len(combined_test_epochs))
        X_test_windows, y_test_windows, _ = epochs_to_windows(
            combined_test_epochs,
            test_groups,
            window_size=config.window_size,
            step_size=config.step_size,
        )
        # Labels are already [0, 1, 2] from epochs_to_windows, no mapping needed

        # Extract features
        test_features = extract_features(X_test_windows, config.fs)

        # Predict (without adaptation first)
        start_eval = time.time() * 1000
        test_preds = clf.predict(test_features)
        end_eval = time.time() * 1000
        test_probs = clf.predict_proba(test_features)

        # Compute metrics
        test_metrics = compile_metrics(
            y_true=y_test_windows,
            y_pred=test_preds,
            y_prob=test_probs,
            timings={
                "train_time": end_train - start_train,
                "infer_latency": (end_eval - start_eval) / max(1, len(y_test_windows)),
                "total_latency": (end_eval - start_eval) / max(1, len(y_test_windows)),
                "filter_latency": filter_obj.get_filter_latency(),
            },
            n_classes=3,
        )
        test_metrics["Dataset"] = "Test (No Adaptation)"
        metrics_table.add_rows([test_metrics])

        # =======================================================================
        # 8. Offline Adaptation Simulation
        # =======================================================================
        if use_adaptation_simulation:
            print("\n" + "=" * 60)
            print("OFFLINE ADAPTATION SIMULATION")
            print("=" * 60)

            # Create a fresh copy for adaptation simulation
            clf_adapt = HybridLDA(
                move_threshold=0.5,
                reg=1e-2,
                shrinkage_alpha=0.1,
                uc_mu=0.4 * 2**-6,
                use_improved_composition=True
            )
            clf_adapt.fit(train_features_all, y_train_windows)

            # Run adaptation simulation
            adapt_results = simulate_offline_adaptation(
                clf_adapt, X_test_windows, y_test_windows, config.fs
            )

            print(f"\n  Results:")
            print(f"    Accuracy (no adaptation):       {adapt_results['acc_no_adapt']:.4f}")
            print(f"    Accuracy (with adaptation):     {adapt_results['acc_with_adapt']:.4f}")
            print(f"    Accuracy (post-adaptation):     {adapt_results['acc_post_adapt']:.4f}")
            print(f"    Updates performed:              {adapt_results['update_stats']['n_updates']}")
            print(f"    Stage A updates (rest/move):    {adapt_results['update_stats']['stage_a']}")
            print(f"    Stage B updates (left/right):   {adapt_results['update_stats']['stage_b']}")

            # Add adapted metrics to table
            adapted_metrics = test_metrics.copy()
            adapted_metrics["Dataset"] = "Test (With Adaptation)"
            adapted_metrics["Acc."] = adapt_results['acc_post_adapt']
            metrics_table.add_rows([adapted_metrics])

        # Display final results
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        metrics_table.display()

        # Save test confusion matrix
        test_cm = confusion_matrix(y_test_windows, test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Rest', 'Left', 'Right'],
                    yticklabels=['Rest', 'Left', 'Right'])
        plt.xlabel('Predicted', fontsize=12, fontweight='bold')
        plt.ylabel('True', fontsize=12, fontweight='bold')
        plt.title('HybridLDA Test - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        test_cm_path = current_wd / "hybrid_lda_test_confusion_matrix.png"
        plt.savefig(test_cm_path, dpi=150)
        print(f"\nTest confusion matrix saved: {test_cm_path}")
        plt.close()

    # ==========================================================================
    # 9. Save the Trained Model
    # ==========================================================================
    model_path = current_wd / "resources" / "models" / "hybrid_lda.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump({
            'model': clf,
            'sfreq': config.fs,
            'n_features': clf.n_features_,
            'stage_info': clf.get_stage_info(),
        }, f)
    print(f"\nHybrid LDA model saved to: {model_path}")

    print("\n" + "=" * 60)
    print("HYBRID LDA OFFLINE EVALUATION COMPLETE!")
    print("=" * 60)

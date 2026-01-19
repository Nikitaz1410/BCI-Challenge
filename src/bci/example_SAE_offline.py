import sys
import pickle
import time

import numpy as np

from pathlib import Path
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold


from bci.loading.bci_config import load_config
from bci.loading.data_acquisition import load_data
from bci.preprocessing.filtering import Filter
from bci.preprocessing.epoching import extract_epochs
from bci.preprocessing.artefact_removal import ArtefactRemoval
from bci.models.SAE import SAEModel
from bci.evaluation.metrics import compute_ece, MetricsTable, compute_itr


# =============================================================================
# Helper Functions
# =============================================================================
def compute_metrics(y_true, y_pred, y_prob, train_time_ms, infer_time_ms, 
                    filter_latency_ms, n_classes):
    """Compute all metrics for a given set of predictions."""
    metrics = {}
    metrics["Acc."] = round(accuracy_score(y_true, y_pred), 4)
    metrics["F1 Score"] = round(f1_score(y_true, y_pred, average="macro"), 4)
    metrics["B. Acc."] = round(balanced_accuracy_score(y_true, y_pred), 4)
    metrics["ECE"] = round(compute_ece(y_true, y_prob, n_bins=10), 4)
    metrics["Brier"] = round(brier_score_loss(y_true, y_prob), 4)
    metrics["Train Time (ms)"] = round(train_time_ms, 2)
    metrics["Infer. Time (ms)"] = round(infer_time_ms, 2)
    metrics["Avg. Filter Latency (ms)"] = round(filter_latency_ms, 2)
    
    itr = compute_itr(
        n_classes=n_classes,
        accuracy=metrics["Acc."],
        time_per_trial=(metrics["Infer. Time (ms)"] + filter_latency_ms) / 1000,
    )
    metrics["ITR (bits/min)"] = round(itr["itr_bits_per_min"], 2)
    
    return metrics


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # TODO: Load configuration
    try:
        config_path = Path.cwd() / "resources" / "configs" / "bci_config.yaml"
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        print("Configuration loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    # INIT OBJECTS AND PARAMETERS
    train_subject_id = 2
    test_subject_id = 42

    filter = Filter(config, online=False)
    ar = ArtefactRemoval()
    clf = SAEModel(classifier="csp-lda")

    metrics_table = MetricsTable()

    # TODO: Import Data
    raw_train, events_train, event_id_train = load_data(
        subject=train_subject_id, config=config
    )
    raw_test, events_test, event_id_test = load_data(
        subject=test_subject_id, config=config
    )

    # TODO: Filter the Data (No Notch for now)
    raw_train = filter.apply_filter_offline(raw_train)
    raw_test = filter.apply_filter_offline(raw_test)

    # TODO: Epoch the Data
    # ATTENTION: DATA LEAKAGE HERE!!! Overlapping windows which are later split in train and val
    train_epochs, train_labels = extract_epochs(
        raw=raw_train, events=events_train, event_id=event_id_train, config=config
    )
    test_epochs, test_labels = extract_epochs(
        raw=raw_test, events=events_test, event_id=event_id_test, config=config
    )

    # =============================================================================
    # Cross-Validation on Training Data
    # =============================================================================
    print("\n" + "="*60)
    print("CROSS-VALIDATION ON TRAINING DATA")
    print("="*60)
    
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_metrics_list = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(train_epochs, train_labels), 1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")
        
        # Split data for this fold
        fold_train_epochs = train_epochs[train_idx]
        fold_train_labels = train_labels[train_idx]
        fold_val_epochs = train_epochs[val_idx]
        fold_val_labels = train_labels[val_idx]
        
        # Remove artifacts on training fold only
        fold_ar = ArtefactRemoval()
        fold_ar.get_rejection_thresholds(fold_train_epochs, config)
        
        # Clean both training and validation folds
        clean_fold_train_epochs, clean_fold_train_labels = fold_ar.reject_bad_epochs(
            fold_train_epochs, fold_train_labels
        )
        clean_fold_val_epochs, clean_fold_val_labels = fold_ar.reject_bad_epochs(
            fold_val_epochs, fold_val_labels
        )
        
        # Train model on clean training fold
        fold_clf = SAEModel(classifier="csp-lda")
        start_train_time = time.time() * 1000
        fold_clf.fit(clean_fold_train_epochs, clean_fold_train_labels)
        end_train_time = time.time() * 1000
        train_time_ms = end_train_time - start_train_time
        
        # Evaluate on validation fold
        start_eval_time = time.time() * 1000
        fold_predictions = fold_clf.predict(clean_fold_val_epochs)
        end_eval_time = time.time() * 1000
        fold_probabilities = fold_clf.predict_proba(clean_fold_val_epochs)
        infer_time_ms = (end_eval_time - start_eval_time) / len(clean_fold_val_labels)
        
        # Compute metrics for this fold
        fold_metrics = compute_metrics(
            y_true=clean_fold_val_labels,
            y_pred=fold_predictions,
            y_prob=fold_probabilities,
            train_time_ms=train_time_ms,
            infer_time_ms=infer_time_ms,
            filter_latency_ms=filter.get_filter_latency(),
            n_classes=len(event_id_train)
        )
        cv_metrics_list.append(fold_metrics)
        print(f"Fold {fold_idx} Accuracy: {fold_metrics['Acc.']:.4f}")
    
    # Compute mean and std of CV metrics
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (Mean ± Std)")
    print("="*60)
    
    cv_mean_metrics = {}
    cv_std_metrics = {}
    metric_keys = cv_metrics_list[0].keys()
    
    for key in metric_keys:
        values = [m[key] for m in cv_metrics_list]
        if key in ["Train Time (ms)", "Infer. Time (ms)", "Avg. Filter Latency (ms)", "ITR (bits/min)"]:
            # For time metrics, compute mean only
            cv_mean_metrics[key] = round(np.mean(values), 2)
            cv_std_metrics[key] = round(np.std(values), 2)
        else:
            # For accuracy/metrics, compute mean and std
            cv_mean_metrics[key] = round(np.mean(values), 4)
            cv_std_metrics[key] = round(np.std(values), 4)
    
    # Create CV summary row
    cv_summary = {}
    for key in metric_keys:
        if key in ["Train Time (ms)", "Infer. Time (ms)", "Avg. Filter Latency (ms)", "ITR (bits/min)"]:
            cv_summary[key] = f"{cv_mean_metrics[key]:.2f} ± {cv_std_metrics[key]:.2f}"
        else:
            cv_summary[key] = f"{cv_mean_metrics[key]:.4f} ± {cv_std_metrics[key]:.4f}"
    cv_summary["Dataset"] = "CV (Training)"
    
    # Add CV results to metrics table
    metrics_table.add_rows([cv_summary])
    metrics_table.display()
    
    # =============================================================================
    # Final Evaluation on Holdout Test Set
    # =============================================================================
    print("\n" + "="*60)
    print("EVALUATION ON HOLDOUT TEST SET")
    print("="*60)
    
    # Remove artifacts from full training data
    ar.get_rejection_thresholds(train_epochs, config)
    clean_train_epochs, clean_train_labels = ar.reject_bad_epochs(
        train_epochs, train_labels
    )
    
    # Also clean test data using the same thresholds
    clean_test_epochs, clean_test_labels = ar.reject_bad_epochs(
        test_epochs, test_labels
    )
    
    # Train the final model on all clean training data
    print("\nTraining final model on all training data...")
    start_train_time = time.time() * 1000
    clf.fit(clean_train_epochs, clean_train_labels)
    end_train_time = time.time() * 1000
    
    # Evaluate on holdout test set
    print("Evaluating on holdout test set...")
    start_eval_time = time.time() * 1000
    test_predictions = clf.predict(clean_test_epochs)
    end_eval_time = time.time() * 1000
    test_probabilities = clf.predict_proba(clean_test_epochs)
    
    # Compute metrics for test set
    test_metrics = compute_metrics(
        y_true=clean_test_labels,
        y_pred=test_predictions,
        y_prob=test_probabilities,
        train_time_ms=end_train_time - start_train_time,
        infer_time_ms=(end_eval_time - start_eval_time) / len(clean_test_labels),
        filter_latency_ms=filter.get_filter_latency(),
        n_classes=len(event_id_train)
    )
    test_metrics["Dataset"] = "Test (Holdout)"
    
    # Add test results to metrics table
    metrics_table.add_rows([test_metrics])
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    metrics_table.display()

    # TODO: Save the Model and other Objects
    model_path = Path.cwd() / "resources" / "models" / "model.pkl"
    clf.save(str(model_path))
    print(f"Model saved to: {model_path}")

    # TODO: Save the Artefact Removal Object
    ar_path = Path.cwd() / "resources" / "models" / "artefact_removal.pkl"
    with open(ar_path, "wb") as f:
        pickle.dump(ar, f)
    print(f"Artefact Removal object saved to: {ar_path}")

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


from bci.loading.bci_config import load_config
from bci.loading.data_acquisition import load_data, extract_baseline
from bci.preprocessing.filtering import Filter
from bci.preprocessing.epoching import extract_epochs
from bci.preprocessing.artefact_removal import ArtefactRemoval
from bci.models.riemann import (
    RiemannianClf,
    recentering,
    compute_covariances,
)
from bci.evaluation.metrics import compute_ece, MetricsTable, compute_itr

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
    clf = RiemannianClf()

    metrics_table = MetricsTable()
    metrics_data = dict()

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
    train_epochs, train_labels = extract_epochs(
        raw=raw_train, events=events_train, event_id=event_id_train, config=config
    )
    test_epochs, test_labels = extract_epochs(
        raw=raw_test, events=events_test, event_id=event_id_test, config=config
    )

    # TODO: Remove Artifacts
    ar.get_rejection_thresholds(train_epochs, config)
    clean_train_epochs, clean_train_labels = ar.reject_bad_epochs(
        train_epochs, train_labels
    )

    # TODO: Extract Features
    train_covs = compute_covariances(clean_train_epochs, cov_est="lwf")
    test_covs = compute_covariances(test_epochs, cov_est="lwf")

    print(f"Train Covs: {train_covs.shape}")
    print(f"Train Labels: {clean_train_labels.shape}")
    print(f"Test Covs: {test_covs.shape}")
    print(f"Test Labels: {test_labels.shape}")

    # TODO: Train the Model
    clf = RiemannianClf()
    start_train_time = time.time() * 1000
    clf.fit(train_covs, clean_train_labels)
    end_train_time = time.time() * 1000

    # TODO: Evaluate the Model
    start_eval_time = time.time() * 1000
    predictions = clf.predict(test_covs)
    end_eval_time = time.time() * 1000
    probabilities = clf.predict_proba(test_covs)

    # TODO: Collect and Compute Metrics -> How should I move this into one function?
    acc = round(accuracy_score(test_labels, predictions), 2)
    metrics_data["Acc."] = acc
    f1 = round(f1_score(test_labels, predictions, average="macro"), 2)
    metrics_data["F1 Score"] = f1
    b_acc = round(balanced_accuracy_score(test_labels, predictions), 2)
    metrics_data["B. Acc."] = b_acc
    ece = round(compute_ece(test_labels, probabilities, n_bins=10), 2)
    metrics_data["ECE"] = ece
    brier = round(brier_score_loss(test_labels, probabilities), 2)
    metrics_data["Brier"] = brier
    train_time = round(end_train_time - start_train_time, 2)
    metrics_data["Train Time (ms)"] = train_time
    infer_time = round((end_eval_time - start_eval_time) / len(test_labels), 2)
    metrics_data["Infer. Time (ms)"] = infer_time
    filter_latency = round(filter.get_filter_latency(), 2)
    metrics_data["Avg. Filter Latency (ms)"] = filter_latency

    itr = compute_itr(
        n_classes=len(event_id_train),
        accuracy=acc,
        time_per_trial=(infer_time + filter_latency) / 1000,
    )
    metrics_data["ITR (bits/min)"] = round(itr["itr_bits_per_min"], 2)

    # Create Offline Metrics Table
    metrics_table.add_rows([metrics_data])
    metrics_table.display()

    # TODO: Save the Model and other Objects
    model_path = Path.cwd() / "resources" / "models" / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to: {model_path}")

    # TODO: Save the Artefact Removal Object
    ar_path = Path.cwd() / "resources" / "models" / "artefact_removal.pkl"
    with open(ar_path, "wb") as f:
        pickle.dump(ar, f)
    print(f"Artefact Removal object saved to: {ar_path}")

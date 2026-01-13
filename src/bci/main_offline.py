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

# Data-loading helpers (physionet + target subject loader)
from bci.Loading.loading import load_physionet_data, load_target_subject_data

# Epoch extraction (preprocessing helper)
from bci.preprocessing.preprocessing import extract_epochs

# Optional: filtering helper and CV helpers
from bci.preprocessing.filtering import Filter
from bci.Training.cv import grouped_kfold_indices
from bci.preprocessing.filtering import Filter
from bci.preprocessing.epoching import extract_epochs
from bci.preprocessing.artefact_removal import ArtefactRemoval
from bci.models.riemann import (
    RiemannianClf,
    recentering,
    compute_covariances,
)
from bci.evaluation.metrics import compute_ece, MetricsTable, compute_itr


from bci.preprocessing.filters import (
    filter_dataset_pair,
    create_filter_from_config,
)
from bci.Training.trainer import run_cross_validation

# =============================================================================
# Main
# =============================================================================


# TODO: Import Data
# Daria START
# NOTE: For Physionet, each output raw = 1 session (for 1 subject) with 3 concatenated runs
# For target subject (Fina), each xdf file is processed as 1 raw = 1 session
# We assign subject IDs to raws.  
root_path = str(Path(__file__).resolve().parents[2])  # BCI-Challange directory
source_path = str(Path(__file__).resolve().parents[2] / "data" / "eeg" / "sub-P999" / "eeg" ) # sub-P999 recordings 

# We need to save and load the data from disk to avoid memory issues
x_raw_train, events_train, event_id_train, sub_ids_train = load_physionet_data(
                                            subjects=list(range(1, 110)), 
                                            root=root_path)

# How the data looks:
# print(f"first 2 elements of x_raw_train: {x_raw_train[0:2]}")
# print(f"element of events_train: {events_train[0]}") # list of arrays of events for each subject
# print(f"event_id_train: {event_id_train}")
# print("first 2 subject IDs:", sub_ids_train[0:2])
# print("\n")

x_raw_test, events_test, event_id_test, sub_ids_test = load_target_subject_data(
                                            root=str(Path(__file__).resolve().parents[2]), 
                                            source_folder=source_path, 
                                            task_type="all", 
                                            limit=None
)

# How the data looks:
# print(f"total {len(x_raw_test)} elements in x_raw_test, first 2 elements of x_raw_test: {x_raw_test[0:2]}")
# print(f"total {len(events_test)} elements in events_test, element of events_test: {events_test[0]}") # list of arrays of events for each subject
# print(f"event_id_test: {event_id_test}")
# print("first 2 subject IDs:", sub_ids_test[0:2], "total", len(sub_ids_test), "subjects")
# print("\n")

#   -> This function should load all usable Finas EEG data
# Daria END

# Amal START
config = load_config()
filt = create_filter_from_config(config, online=False)
# TODO: Filter the Data (No Notch for now)

#change: perform offline filtering for train and test using the preprocessing helper
x_filtered_train, x_filtered_test, filter_latency_ms = filter_dataset_pair(
    x_raw_train, x_raw_test, config
)

# TODO: Epoch the Data
# ATTENTION: DATA LEAKAGE HERE!!! Overlapping windows which are later split in train and val

train_epochs, train_labels = extract_epochs(
    raw=x_filtered_train, events=y_train, event_id=sessions_id_train, config=config
)
test_epochs, test_labels = extract_epochs(
    raw=x_filtered_test, events=y_test, event_id=sessions_id_test, config=config
)

# Begin Cross Validation just on the training data and test on finas test data only in the end 

#change: run grouped CV and training in trainer.run_cross_validation to keep main minimal
cv_out = run_cross_validation(
    train_epochs,
    train_labels,
    sessions_id_train,
    config,
    n_splits=int(config.get("n_splits", 5)),
)

# Original TODO block (kept for traceability):
# 1. Split the training data into k folds
# 2. For each fold, train the model on the training data and test on the validation data
# 3. Compute the metrics for each fold
# 4. Compute the average metrics across all folds
# 5. Compute the final metrics on the test data
# 6. Print the metrics
# 7. Save the model


# (optional) Normalize the data
#train_epochs = recentering(train_epochs)
#test_epochs = recentering(test_epochs)

# TODO: Remove Artifacts
#change: Delegate AutoReject + windowing to preprocessing.autoreject_and_window
from bci.preprocessing.preprocessing import autoreject_and_window

# Fit AutoReject on training epochs and extract windows
clean_train_epochs, X_train, y_train, sessions_id_train, ar = autoreject_and_window(
    train_epochs, train_labels, sessions_id_train, config, ar=None, fit_ar=True
)

# Apply the fitted AutoReject to test epochs and extract windows
clean_test_epochs, X_test, y_test, sessions_id_test, _ = autoreject_and_window(
    test_epochs, test_labels, sessions_id_test, config, ar=ar, fit_ar=False
)

# @Amal please verify that sessions_id_train and sessions_id_test align with downstream assumptions
# Amal END

# Nikita START
# TODO: Extract Features
model.fit(clean_train_epochs, clean_train_labels)
model.predict(test_epochs)

# Nikita END

#Iustin START
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
#change: use estimated offline latency from preprocessing helper (filter_dataset_pair)
filter_latency = round(filter_latency_ms, 2)
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
model.save(model_path)

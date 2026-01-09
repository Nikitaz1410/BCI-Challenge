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


# TODO: Import Data
# Daria START
x_raw_train, y_train, sessions_id_train = load_data("train")
#   -> This function should load all available EEGBCI data 
x_raw_test, y_test, sessions_id_test = load_data("test")
#   -> This function should load all usable Finas EEG data
# Daria END

# Amal START
# TODO: Filter the Data (No Notch for now)
x_filtered_train = filter.apply_filter_offline(x_raw_train)
x_filtered_test = filter.apply_filter_offline(x_raw_test)

# TODO: Epoch the Data
train_epochs, train_labels = extract_epochs(
    raw=x_filtered_train, events=y_train, event_id=sessions_id_train, config=config
)
test_epochs, test_labels = extract_epochs(
    raw=x_filtered_test, events=y_test, event_id=sessions_id_test, config=config
)

# Begin Cross Validation just on the training data and test on finas test data only in the end

# (optional) Normalize the data
train_epochs = recentering(train_epochs)
test_epochs = recentering(test_epochs)

# TODO: Remove Artifacts
ar.get_rejection_thresholds(train_epochs, config)
clean_train_epochs, clean_train_labels = ar.reject_bad_epochs(
    train_epochs, train_labels
)
clean_test_epochs, clean_test_labels = ar.reject_bad_epochs(
    test_epochs, test_labels
)
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
model.save(model_path)

# Pull the chunks of data from the stream
# Buffer the data to form epochs
# Apply preprocessing (filtering, artifact removal) on the epochs
# Compute covariance matrices for the epochs
# Apply Riemannian geometry-based classification on the covariance matrices
# Output the classification results in real-time

import sys
import time
import pickle
import numpy as np
from pathlib import Path

from pylsl import StreamInlet, resolve_streams

from bci.Loading.bci_config import load_config
from bci.Preprocessing.filters import Filter
from bci.Preprocessing.artefact_removal import ArtefactRemoval
from bci.Utils.utils import choose_model

if __name__ == "__main__":
    # Init the Objects

    # TODO: Load configuration
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

    model_path = Path.cwd() / "resources" / "models" / "model.pkl"
    artefact_rejection_path = (
        Path.cwd() / "resources" / "models" / "artefact_removal.pkl"
    )

    filter = Filter(config, online=True)
    ar = pickle.load(open(artefact_rejection_path, "rb"))
    model_args = {
        "cov_est": "lwf"
    }  # args for the model chooser (what the model constructor needs)
    clf = choose_model(config.model)
    clf = clf.load(model_path)

    buffer = np.zeros((len(config.channels), int(config.window_size)), dtype=np.float32)
    label_buffer = np.zeros((1, int(config.window_size)), dtype=np.int32)

    avg_time_per_classification = 0.0
    number_of_classifications = 0

    total_fails = 0
    total_successes = 0
    total_trials = 0

    total_predictions = 0
    total_rejected = 0

    probability_threshold = 0.6  # threshold for accepting a prediction

    markers = {
        0: "unknown",
        1: "rest",
        2: "left_hand",
        3: "right_hand",
    }

    print("Initializing preprocessing and model objects completed!")

    # TODO: Find the EEG stream from LSL and establish connection
    print("Looking for an EEG and Markers streams...")
    streams = resolve_streams(wait_time=5.0)

    eeg_streams = [s for s in streams if s.type() == "EEG"]
    label_streams = [s for s in streams if s.type() == "Markers"]

    # How does this affect?
    inlet = StreamInlet(eeg_streams[0], max_chunklen=32)
    inlet_labels = StreamInlet(label_streams[0], max_chunklen=32)

    print("Starting to read data from the EEG stream...")

    while True:
        # TODO: Check the frequency at which the data is pulled
        try:
            start_classification_time = time.time() * 1000  # in milliseconds
            sample, timestamp = inlet.pull_chunk()
            labels, label_timestamp = inlet_labels.pull_chunk()
            crt_label = None
            if len(sample) == 0 or len(labels) == 0:
                continue
            # Update the buffer with new samples
            else:
                sample = np.array(sample).T  # shape (n_channels, n_samples)
                labels = np.array(labels).T  # shape (1, n_samples)
                n_new_samples = sample.shape[1]
                n_new_labels = labels.shape[1]

                # Shift the buffer to the left
                buffer = np.roll(buffer, -n_new_samples, axis=1)
                label_buffer = np.roll(label_buffer, -n_new_samples, axis=1)

                # Add new samples to the end of the buffer
                buffer[:, -n_new_samples:] = sample
                label_buffer[:, -n_new_labels:] = labels

                # Extract the current label (most present in the buffer)
                unique, counts = np.unique(label_buffer, return_counts=True)
                label_counts = dict(zip(unique, counts))
                crt_label = max(label_counts, key=label_counts.get)
                print(
                    f"Current label: {crt_label} - {markers.get(crt_label, 'unknown')}"
                )

            # TODO: Filter the data
            filtered_data = filter.apply_filter_online(buffer)

            # TODO: Artifact Rejection: Check if it is an artifact => Skipped for now because it rejects only

            # TODO: Create the features and classify
            probability = clf.predict_proba(filtered_data)

            if probability is None:
                pass  # error in prediction
            prediction = np.argmax(probability, axis=1)[0]
            print(
                f"Predicted class: {prediction} - {markers.get(prediction, 'unknown')} with probability {probability[0][prediction]:.4f}"
            )
            total_predictions += 1
            if probability[0][prediction] < probability_threshold:
                total_rejected += 1
            else:
                total_successes += int(prediction == crt_label)
                total_fails += int(prediction != crt_label)

            # TODO: Integrate the transfer function

            end_classification_time = time.time() * 1000  # in milliseconds
            avg_time_per_classification += (
                end_classification_time - start_classification_time
            )
        except KeyboardInterrupt:
            print("Stopping the online processing.")
            # Add the current evaluations?
            print(
                f"Avg time per loop: {avg_time_per_classification / max(1, number_of_classifications)} ms"
            )
            print(
                f"Total Predictions: {total_predictions}, Rejected: {total_rejected},  Successes: {total_successes}, Fails: {total_fails}"
            )
            break

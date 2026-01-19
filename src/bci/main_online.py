# Pull the chunks of data from the stream
# Buffer the data to form epochs
# Apply preprocessing (filtering, artifact removal) on the epochs
# Compute covariance matrices for the epochs
# Apply Riemannian geometry-based classification on the covariance matrices
# Output the classification results in real-time

import pickle
import socket
import sys
import time
from pathlib import Path

import numpy as np
from pylsl import StreamInlet, resolve_streams

from bci.preprocessing.filters import Filter
from bci.Transfer.transfer import BCIController
from bci.Utils.bci_config import load_config
from bci.Utils.utils import choose_model

if __name__ == "__main__":
    # Initialize the Objects

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

    model_path = current_wd / "resources" / "models" / "model.pkl"
    artefact_rejection_path = (
        current_wd / "resources" / "models" / "artefact_removal.pkl"
    )

    filter = Filter(config, online=True)
    ar = pickle.load(open(artefact_rejection_path, "rb"))
    model_args = {
        "cov_est": "lwf"
    }  # args for the model chooser (what the model constructor needs)
    clf = choose_model(config.model)
    clf = clf.load(model_path)

    # Transfer Function
    controller = BCIController(config)

    buffer = np.zeros((len(config.channels), int(config.window_size)), dtype=np.float32)
    label_buffer = np.zeros((1, int(config.window_size)), dtype=np.int32)

    avg_time_per_classification = 0.0
    number_of_classifications = 0

    total_fails = 0
    total_successes = 0

    total_predictions = 0
    total_rejected = 0

    probability_threshold = 0.6  # threshold for accepting a prediction

    markers = {
        0: "unknown",
        1: "rest",
        2: "left_hand",
        3: "right_hand",
    }

    # Add stream finding logic based on mode

    print("Initializing preprocessing and model objects completed!")

    # Find the EEG stream from LSL and establish connection
    print("Looking for EEG and Markers streams...")
    streams = resolve_streams(wait_time=5.0)

    eeg_streams = [s for s in streams if s.type() == "EEG"]
    if config.online == "dino":
        label_streams = [
            s
            for s in streams
            if s.type() == "Markers" and s.name() == "MyDinoGameMarkerStream"
        ]
    else:
        label_streams = [
            s for s in streams if s.type() == "Markers" and s.name() == "Labels_Stream"
        ]

    if not eeg_streams or not label_streams:
        print("❌ Could not find EEG or Markers streams.")
        sys.exit(1)

    inlet = StreamInlet(eeg_streams[0], max_chunklen=32)
    inlet_labels = StreamInlet(label_streams[0], max_chunklen=32)

    print("Starting to read data from the EEG stream...")
    print(f"Reading Labels from {label_streams[0].name()}")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        while True:
            try:
                start_classification_time = time.time() * 1000  # in milliseconds
                eeg_chunk, timestamp = inlet.pull_chunk()
                labels_chunk, label_timestamp = inlet_labels.pull_chunk()
                crt_label = None

                # Check if sample and labels are valid and non-empty
                if eeg_chunk:
                    # Convert to numpy arrays and transpose to (n_channels, n_samples)
                    eeg_chunk = np.array(eeg_chunk).T  # shape (n_channels, n_samples)
                    n_new_samples = eeg_chunk.shape[1]
                    # Safety: If new data is larger than the buffer, just take the end of it
                    if n_new_samples >= config.window_size:
                        buffer = eeg_chunk[:, -config.window_size :]

                    # Update the buffers with the new chunks of data
                    buffer[:, :-n_new_samples] = buffer[:, n_new_samples:]
                    buffer[:, -n_new_samples:] = eeg_chunk

                if labels_chunk:
                    labels_chunk = np.array(labels_chunk).T  # shape (1, n_samples)

                    n_new_labels = labels_chunk.shape[1]

                    if n_new_labels >= config.window_size:
                        label_buffer = labels_chunk[:, -config.window_size :]

                    label_buffer[:, :-n_new_labels] = label_buffer[:, n_new_labels:]
                    label_buffer[:, -n_new_labels:] = labels_chunk

                # Extract the current label (most present in the buffer)
                # TODO: check this for the dino game -> Might need mapping
                unique, counts = np.unique(label_buffer, return_counts=True)
                if len(unique) > 0:
                    label_counts = dict(zip(unique, counts))
                    crt_label = max(label_counts, key=lambda k: label_counts[k])
                    print(
                        f"Current label: {crt_label} - {markers.get(crt_label, 'unknown')}"
                    )
                else:
                    crt_label = 0  # fallback to unknown

                # Filter the data
                filtered_data = filter.apply_filter_online(buffer)

                # Artifact Rejection: Skipped for now

                # Create the features and classify
                probabilities = clf.predict_proba(filtered_data)

                if probabilities is None:
                    print("Warning: Model returned None for probability.")
                    continue  # skip this iteration

                controller.send_command(probabilities, sock)

                prediction = np.argmax(
                    probabilities, axis=1
                )[
                    0
                ]  # To account to the fact that the classifier was trained with labels 1,2,3
                print(
                    f"Predicted class: {prediction+1} - {markers.get(prediction+1, 'unknown')} with probability {probabilities[0][prediction]:.4f}"
                )
                total_predictions += 1
                if probabilities[0][prediction] < probability_threshold:
                    total_rejected += 1
                else:
                    total_successes += int(prediction == crt_label)
                    total_fails += int(prediction != crt_label)

                number_of_classifications += 1

                end_classification_time = time.time() * 1000  # in milliseconds
                avg_time_per_classification += (
                    end_classification_time - start_classification_time
                )
            except KeyboardInterrupt:
                print("Stopping the online processing.")
                print(
                    f"Avg time per loop: {avg_time_per_classification / max(1, number_of_classifications):.2f} ms"
                )
                print(
                    f"Total Predictions: {total_predictions}, Rejected: {total_rejected},  Successes: {total_successes}, Fails: {total_fails}"
                )
                break

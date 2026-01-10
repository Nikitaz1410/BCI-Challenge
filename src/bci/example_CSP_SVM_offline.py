import sys
import time
import pickle
import numpy as np
from pathlib import Path

# Ensure the package root is on sys.path when running this file directly
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pylsl import StreamInlet, resolve_streams

from bci.loading.bci_config import load_config
from bci.preprocessing.filtering import Filter
from bci.preprocessing.epoching import extract_epochs
from bci.preprocessing.artefact_removal import ArtefactRemoval
from bci.models.riemann import (
    RiemannianClf,
    recentering,
    compute_covariances,
)
from bci.evaluation.metrics import compute_ece, MetricsTable, compute_itr

if __name__ == "__main__":
    # Init the Objects

    # TODO: Load configuration
    try:
        config_path = Path.cwd() / "resources" / "configs" / "bci_config.yaml"
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        print("Configuration loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    model_path = Path.cwd() / "resources" / "models" / "model.pkl"
    artefact_rejection_path = (
        Path.cwd() / "resources" / "models" / "artefact_removal.pkl"
    )

    filter = Filter(config, online=True)
    ar = pickle.load(open(artefact_rejection_path, "rb"))
    clf = pickle.load(open(model_path, "rb"))
    buffer = np.zeros((len(config.channels), int(config.window_size)), dtype=np.float32)

    avg_time_per_classification = 0.0
    number_of_classifications = 0

    total_fails = 0
    total_successes = 0
    total_trials = 0

    print("Initializing preprocessing and model objects completed!")

    # TODO: Find the EEG stream from LSL and establish connection
    print("Looking for an EEG stream...")
    streams = resolve_streams(wait_time=5.0)
    print("EEG stream found.")
    print(streams)
    inlet = StreamInlet(streams[0])
    inlet_labels = StreamInlet(streams[1])
    print("Starting to read data from the EEG stream...")
    while True:
        # TODO: Check the frequency at which the data is pulled
        try:
            start_classification_time = time.time() * 1000  # in milliseconds
            sample, timestamp = inlet.pull_chunk()
            if len(sample) == 0:
                continue
            # Update the buffer with new samples
            else:
                sample = np.array(sample).T  # shape (n_channels, n_samples)
                # print(f"New sample shape: {sample.shape}")
                # Add the new samples to the buffer at the end and remove the beginning
                n_new_samples = sample.shape[1]
                buffer = np.roll(buffer, -n_new_samples, axis=1)
                buffer[:, -n_new_samples:] = sample

            # TODO: Filter the data
            filtered_data = filter.apply_filter_online(buffer)

            # TODO: Artifact Rejection: Check if it is an artifact

            # TODO: Create the features and classify
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
            break

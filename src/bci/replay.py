# A script which sense chunks of data to the lsl stream that can be picked up by our BCI Software
# Goal: Simulate a real-time data stream for testing and development purposes

import sys
import time
from pathlib import Path

import numpy as np
from pylsl import StreamInfo, StreamOutlet

from bci.loading.bci_config import load_config
from bci.loading.data_acquisition import load_data


def replay_data_as_lsl(
    eeg_data: np.ndarray, labels: np.ndarray, sfreq: float, chunk_size: int = 32
):
    """
    Replay EEG data and associated labels as LSL streams.

    Parameters:
    - eeg_data: np.ndarray
        The EEG data to be streamed, shape (n_channels, n_samples).
    - labels: np.ndarray
        The labels associated with the EEG data, shape (n_samples,).
    - sfreq: float
        The sampling frequency of the EEG data.
    - chunk_size: int
        The number of samples to send in each chunk.
    """

    n_channels, n_samples = eeg_data.shape

    if labels.shape[0] != n_samples:
        raise ValueError(
            "labels must have the same number of samples as eeg_data (axis=1)"
        )

    # Create LSL stream info and outlet for EEG data
    info_eeg = StreamInfo(
        "EEG_Stream", "EEG", n_channels, sfreq, "float32", "myuid34234"
    )
    outlet_eeg = StreamOutlet(info_eeg)

    # Create LSL stream info and outlet for labels
    info_labels = StreamInfo(
        "Labels_Stream", "Markers", 1, sfreq, "int32", "myuid_labels_34234"
    )
    outlet_labels = StreamOutlet(info_labels)

    print("Starting LSL streams (EEG and Labels)...")

    # Calculate the time interval between chunks
    chunk_interval = chunk_size / sfreq

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk = eeg_data[:, start_idx:end_idx]
        label_chunk = labels[start_idx:end_idx]

        # Send the chunk to the LSL EEG stream
        outlet_eeg.push_chunk(chunk.T.tolist())

        # Send the corresponding labels as a chunk (as a list of lists, each with one label)
        outlet_labels.push_chunk([[int(l)] for l in label_chunk])

        # Wait for the appropriate time before sending the next chunk
        time.sleep(chunk_interval)

    print("Finished streaming EEG data and labels.")


# Example usage:

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

    # Load data file and prepare data
    test_subject_id = 42
    raw_data, events, event_id = load_data(subject=test_subject_id, config=config)
    eeg_data = raw_data.get_data()  # shape (n_channels, n_samples)
    labels = np.zeros(eeg_data.shape[1], dtype=int)

    for event in events:
        labels[event[0]] = event[2]

    print(f"EEG Data shape: {eeg_data.shape}, Labels shape: {labels.shape}")

    # Replay the data and labels as LSL streams
    replay_data_as_lsl(eeg_data, labels, config.fs, chunk_size=32)

# A script which sense chunks of data to the lsl stream that can be picked up by our BCI Software
# Goal: Simulate a real-time data stream for testing and development purposes

import sys
import time
from pathlib import Path

# Add src directory to Python path to allow imports
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent  # Go from src/bci/replay.py to src/
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
from pylsl import StreamInfo, StreamOutlet

from bci.loading.loading import load_physionet_data, load_target_subject_data
from bci.utils.bci_config import load_config


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
    # Load the config file
    # Detect project root: handle both workspace root and BCI-Challenge subdirectory
    # Script is at: [workspace]/BCI-Challenge/src/bci/replay.py
    script_dir = Path(__file__).parent.parent.parent  # Goes up 3 levels from script
    
    # Check if we're in a BCI-Challenge subdirectory (workspace structure)
    # The script is at: BCI-Challenge/src/bci/replay.py
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
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    if "Phy" in config.replay_subject_id:
        s_id = int(config.replay_subject_id.split("-")[1])
        # Load training data from Physionet
        raw, events, event_id, sub_ids, train_physionet_filenames = load_physionet_data(
            subjects=[s_id], root=current_wd, channels=config.channels
        )

        print(f"Loaded subject {s_id} from Physionet for training.")
    else:
        # Load target subject data for testing
        test_data_source_path = (
            current_wd / "data" / "eeg" / config.replay_subject_id
        )  # Path can be defined in config file
        # NOTE: To load all target subject data, you need to have "sub" folder in both test_data_source_path with all subject data files

        test_data_target_path = (
            current_wd / "data" / "datasets" / config.replay_subject_id
        )  # Path can be defined in config file

        raw, events, event_id, sub_ids = load_target_subject_data(
            root=current_wd,
            source_path=test_data_source_path,
            target_path=test_data_target_path,
        )

    # Extract EEG data and labels
    # TODO: Handle multiple sessions if needed -> For now only one is considered
    eeg_data = raw[0].get_data()  # shape (n_channels, n_samples)

    # TODO: Check if this is really correct
    labels = np.zeros(eeg_data.shape[1], dtype=int)

    # Assign labels based on events
    for event in events[0]:
        sample_idx = event[0]
        label = event[2]
        labels[sample_idx] = label

    # Populate labels for intervals between events
    for i in range(len(events[0]) - 1):
        start_sample = events[0][i][0]
        end_sample = events[0][i + 1][0]
        label = events[0][i][2]
        labels[start_sample:end_sample] = label

    print(np.unique(labels))
    print(f"EEG data shape: {eeg_data.shape}, Labels shape: {labels.shape}")

    # Replay the data and labels as LSL streams
    replay_data_as_lsl(eeg_data, labels, config.fs, chunk_size=32)

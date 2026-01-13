import sys
import mne

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def extract_epochs(raw, events, event_id, config):
    epochs = mne.Epochs(
        raw,
        events,
        event_id=dict(rest=1, left=2, right=3),
        tmin=1.0,
        tmax=3.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    print(f"Number of rest trials: {epochs["rest"].get_data().shape}")
    print(f"Number of left trials: {epochs["left"].get_data().shape}")
    print(f"Number of right trials: {epochs["right"].get_data().shape}")

    epochs_data = epochs.get_data()
    print(f"Total MI trials shape: {epochs_data.shape}")

    epochs_labels = epochs.events[:, 2] - 1  # Convert to 0-Rest, 1-Left, 2-Right

    # Extract overlapping windows from the trials
    # This is done because in real-time BCI, we need to clasify on a shorter window so as to reduce latency
    # Also done to increase the number of samples for training
    # Leakage dealt with in the train-test split by making sure trials are not split
    windowed_epochs = []
    windowed_labels = []

    for i, eeg in enumerate(epochs_data):
        windows = extract_overlapping_windows(
            eeg, window_size=config.window_size, step_size=config.step_size
        )
        windowed_epochs.append(windows)
        windowed_labels.extend([epochs_labels[i]] * windows.shape[0])

    windowed_epochs = np.concatenate(windowed_epochs, axis=0)
    windowed_labels = np.array(windowed_labels)

    print(f"Windowed epochs shape: {windowed_epochs.shape}")

    return windowed_epochs, windowed_labels


def do_grouped_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int = 42,
    test_size: float = 0.15,
) -> dict:
    """
    Perform grouped split into train and test sets.
    Ensures that groups are not split across sets (no leakage).

    If test_size is 0, returns all data as training data (no split).

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label array.
        groups (np.ndarray): Group identifiers (e.g., trial_ids).
        random_state (int): Seed for reproducibility.
        test_size (float): Proportion of groups for the test set. Set to 0 for 100% train.

    Returns:
        dict: Dictionary containing split arrays (x_train, y_train, etc.)
              and their corresponding original indices (train_idx, etc.).
    """

    # Check for the "no split" case (100% Training Data)
    if test_size <= 0:
        n_samples = len(X)
        train_idx = np.arange(n_samples)
        test_idx = np.array([], dtype=int)  # Empty index array

    else:
        # Standard Split: Separate Test set from Train set
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        # split returns indices relative to the array passed to it
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # Create the Test set (will be empty arrays if test_size=0)
    X_test = X[test_idx]
    y_test = y[test_idx]
    groups_test = groups[test_idx]

    # Create the Train set
    X_train = X[train_idx]
    y_train = y[train_idx]
    groups_train = groups[train_idx]

    return {
        "x_train": X_train,
        "y_train": y_train,
        "train_idx": train_idx,
        "groups_train": groups_train,
        "x_test": X_test,
        "y_test": y_test,
        "test_idx": test_idx,
        "groups_test": groups_test,
    }


#change: moved window extraction to bci.preprocessing.windows
# Re-export to preserve API used by older code
from bci.preprocessing.windows import extract_overlapping_windows

# Original implementation (commented for traceability):
# def extract_overlapping_windows(eeg, window_size=250, step_size=16):
#     n_channels, n_samples = eeg.shape
#
#     window_length_samples = int(window_size)
#     window_shift_samples = int(step_size)
#
#     nwindows = int((n_samples - window_length_samples) / (window_shift_samples)) + 1
#
#     window_starts = np.arange(
#         0, n_samples - window_length_samples + 1, window_shift_samples
#     ).astype(int)
#     window_ends = window_starts + window_length_samples
#
#     windows = np.zeros((nwindows, n_channels, window_length_samples))
#
#     # Extract the windows
#     for window_id in range(nwindows):
#         start = window_starts[window_id]
#         end = window_ends[window_id]
#
#         window = eeg[:, start:end]
#         windows[window_id, :, :] = window
#
#     return windows


# ==========================================
# PYTESTS
# ==========================================


def test_split_shapes_and_types():
    """Check that output is a dictionary with correct types and non-empty arrays."""
    n_samples = 100
    X = np.random.rand(n_samples, 5)
    y = np.random.randint(0, 2, size=n_samples)
    groups = np.random.randint(0, 10, size=n_samples)  # 10 groups

    splits = do_grouped_train_test_split(X, y, groups, test_size=0.2)

    assert isinstance(splits, dict)
    assert len(splits["train_idx"]) > 0
    assert len(splits["test_idx"]) > 0

    # Check that keys for validation do not exist
    assert "x_val" not in splits
    assert "val_idx" not in splits

    # Check that X splits match indices
    np.testing.assert_array_equal(splits["x_train"], X[splits["train_idx"]])
    np.testing.assert_array_equal(splits["x_test"], X[splits["test_idx"]])


def test_no_group_leakage():
    """Ensure that groups are strictly separated between train and test."""
    n_samples = 200
    X = np.zeros((n_samples, 1))
    y = np.zeros(n_samples)
    # 20 distinct groups
    groups = np.repeat(np.arange(20), 10)

    splits = do_grouped_train_test_split(X, y, groups, test_size=0.2)

    groups_train = set(groups[splits["train_idx"]])
    groups_test = set(groups[splits["test_idx"]])

    # Check intersection is empty
    assert groups_train.isdisjoint(groups_test), "Train and Test groups overlap"


def test_index_integrity():
    """
    Verify that the union of train and test indices covers all samples
    and that indices map to correct values.
    """
    n_samples = 50
    X = np.arange(n_samples).reshape(-1, 1)
    y = np.zeros(n_samples)
    groups = np.random.randint(0, 5, size=n_samples)

    splits = do_grouped_train_test_split(X, y, groups, test_size=0.2)

    # Check coverage
    all_indices = np.concatenate([splits["train_idx"], splits["test_idx"]])
    all_indices.sort()
    np.testing.assert_array_equal(all_indices, np.arange(n_samples))

    # Check values
    recovered_train = X[splits["train_idx"]].flatten()
    actual_train = splits["x_train"].flatten()
    np.testing.assert_array_equal(recovered_train, actual_train)


if __name__ == "__main__":
    try:
        test_split_shapes_and_types()
        test_no_group_leakage()
        test_index_integrity()
        print("All tests passed successfully!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

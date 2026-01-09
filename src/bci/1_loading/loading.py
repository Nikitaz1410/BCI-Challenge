
import warnings
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import scipy
import pyxdf

import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import mne
from mne import Epochs, pick_types
from mne.io import concatenate_raws, read_raw_edf

import moabb
from moabb.datasets import PhysionetMI

from autoreject import AutoReject

import pyxdf

# Add the root directory to the Python path
root_dir = str(Path(__file__).resolve().parents[3])  # BCI-Challange directory
parent_of_root = str(Path(__file__).resolve().parents[4])  # one level above

sys.path.append(root_dir)
sys.path.append(parent_of_root)

sys.path.append("..")
from src.bci.utils.helper import get_raw_offline, get_epochs


moabb.set_log_level("info")
warnings.filterwarnings("ignore")


# CHECK COMMON CHANNELS BETWEEN SUBJECTS P999 AND P554: JUST TO VERIFY HOW MANY CHANNELS WE CAN USE

p999_file = Path(parent_of_root) / "data" / "sub-P999" / "eeg" / "sub-P999_ses-S002_task-arrow_run-001_eeg.xdf"
p554_file = Path(parent_of_root) / "data" / "sub-P554" / "eeg" / "sub-P554_ses-S002_task-Default_run-001_eeg.xdf"

##############################
# Extract channels directly from info stored in the XDF file (reference: helper.py module provided by BCI practical) 
###############################
streams, header = pyxdf.load_xdf(p999_file, verbose=False)

event_channel = None
markers = []

pupil_capture_channel, pupil_capture_fixations_channel, pupil_channel = (
        None,
        None,
        None,
    )
eeg_channel = None

# Iterate through streams to find desired indices
for i, stream in enumerate(streams):
    name = stream["info"]["name"][0]  # name of the stream
    # print(name)
    if name == "EEG - Impedances":
        _index_impedance = i  # impedance stream
    elif name == "EEG":
        eeg_channel = i  # eeg stream
    elif name == "pupil_capture_pupillometry_only":
        pupil_channel = i  # pupil stream
    elif name == "pupil_capture":
        pupil_capture_channel = i  # pupil stream
    elif name == "pupil_capture_fixations":
        pupil_capture_fixations_channel = i
    else:
        event_channel = i  # markers stream

    # Get the channel labels
channels = streams[eeg_channel]["info"]["desc"][0]["channels"][0]["channel"]
p999_channels = [channel["label"][0] for channel in channels]

# Channels manually set for P554 (g.Nautilus) got while using "original" get_raw_offline function from helper.py
_, _, p554_channels = get_raw_offline(p554_file)

print(f"Number of channels for P999: {len(p999_channels)}, channels: {p999_channels}")
print(f"Number of channels for P554: {len(p554_channels)}, channels: {p554_channels}")

common_channels = list(set(p999_channels).intersection(set(p554_channels)))
print(f"Common channels between P999 and P554: {common_channels}")
print(f"Number of common channels: {len(common_channels)}")
# ONLY 13 CHANNELS ARE COMMON: ['C4', 'Fp2', 'C3', 'T7', 'T8', 'F4', 'Fp1', 'P3', 'Cz', 'Fz', 'F3', 'P4', 'Pz']


# # CSP Feature Extractor class (from A4_1)
# class CSPFeatureExtractor:
#     def __init__(self, n_components: int = 4, reg = None):
#         self.n_components = n_components
#         self.reg = reg
#         self.csp = mne.decoding.CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)

#     def fit(self, data: np.ndarray, labels: np.ndarray):
#         """Fit CSP on raw windowed trials (n_windows, n_channels, n_samples)"""
#         # MNE CSP expects raw data, not covariance matrices
#         self.csp.fit(data, labels)
#         return self

#     def transform(self, data: np.ndarray) -> np.ndarray:
#         """Transform raw windowed trials to CSP features"""
#         return self.csp.transform(data)
    
#     def fit_transform(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
#         """Fit and transform in one step"""
#         self.fit(data, labels)
#         return self.transform(data)
    

# ==========================================
# From bci_processing.py (Assignment 5)
# ==========================================
# Extract overlapping time windows of the trials for feature extraction
def extract_overlapping_windows(eeg, window_size=250, step_size=16):
    n_channels, n_samples = eeg.shape

    window_length_samples = int(window_size)
    window_shift_samples = int(step_size)

    nwindows = int((n_samples - window_length_samples) / (window_shift_samples)) + 1

    window_starts = np.arange(
        0, n_samples - window_length_samples + 1, window_shift_samples
    ).astype(int)
    window_ends = window_starts + window_length_samples

    windows = np.zeros((nwindows, n_channels, window_length_samples))

    # Extract the windows
    for window_id in range(nwindows):
        start = window_starts[window_id]
        end = window_ends[window_id]

        window = eeg[:, start:end]
        windows[window_id, :, :] = window

    return windows

# ==========================================
# from t3_extract_features.py (Assignment 5)
# ==========================================
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

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label array.
        groups (np.ndarray): Group identifiers (e.g., trial_ids).
        random_state (int): Seed for reproducibility.
        test_size (float): Proportion of groups for the test set.

    Returns:
        dict: Dictionary containing split arrays (x_train, y_train, etc.)
              and their corresponding original indices (train_idx, etc.).
    """

    # Single Split: Separate Test set from Train set
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # split returns indices relative to the array passed to it
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # Create the Test set
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



def do_grouped_cv_splits(X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> dict: 
    """
    Perform grouped split into train and val sets for cross-validation.
    Ensures that groups are not split across sets (no leakage).

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label array.
        groups (np.ndarray): Group identifiers (e.g., trial_ids).
        n_splits (int): Number of splits (folds).

    Returns:
        dict: Dictionary containing split arrays (x_train, y_train, etc.)
              and their corresponding original indices (train_idx, etc.).
    """
    
    # Initialize GroupKFold with n_splits folds
    group_kfold = GroupKFold(n_splits=n_splits)

    # Store the splits
    cv_splits = {}

    # Store the group indices of splits
    train_val_idxs = []
    for train_idx, val_idx in group_kfold.split(X, y, groups=groups):
        train_val_idxs.append((train_idx, val_idx))

    for fold, (train_idx, val_idx) in enumerate(train_val_idxs):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train = groups[train_idx]
        groups_val = groups[val_idx]
        cv_splits[f"fold{fold}"] = {
                            "x_train": X_train,
                            "y_train": y_train,
                            "train_idx": train_val_idxs[fold][0],
                            "groups_train": groups_train,
                            "x_val": X_val,
                            "y_val": y_val,
                            "val_idx": train_val_idxs[fold][1],
                            "groups_val": groups_val

                        }
    return cv_splits


# ==========================================
# LOADING PHYSIONET DATA
# ==========================================
print("\n\
------------------------------------------\n\
Loading Physionet data\n\
------------------------------------------")

dataset = PhysionetMI(imagined=True, executed=False)
dataset.feet_runs = []
all_subjects = dataset.subject_list
remove = [88, 92, 100]      # despite report, subjects 88, 92 and 100 were sampled with 128 Hz
subjects = [x for x in all_subjects if x not in remove]

subjects = all_subjects[1:2] # TODO: Fix code crushes if performing autoreject on 5+ subjects
# print(dataset.events)  {'left_hand': 2, 'right_hand': 3, 'feet': 5, 'hands': 4, 'rest': 1}

channels = [
        "Fp1",
        "Fp2",
        "T8",
        "F4",
        "Fz",
        "F3",
        "T7",
        "C4",
        "Cz",
        "C3",
        "P4",
        "Pz",
        "P3",
        "PO8",
        "Oz",
        "PO7",
    ]

raw_dict = dataset.get_data(subjects=subjects)
''' E.g. for subject 2:
{2: 
    {'0': 
        {'0': <RawEDF | S002R04.edf, 65 x 19680 (123.0 s), ~9.8 MiB, data loaded>, 
         '1': <RawEDF | S002R08.edf, 65 x 19680 (123.0 s), ~9.8 MiB, data loaded>, 
         '2': <RawEDF | S002R12.edf, 65 x 19680 (123.0 s), ~9.8 MiB, data loaded>
        }
    }
}
'''
raws_concatenated = {}

for subject in subjects:
    raw_objects = []
    for session in raw_dict[subject]:   # subject level
        for run in raw_dict[subject][str(session)]:  # run level
            raw = raw_dict[subject][str(session)][str(run)]
            raw_objects.append(raw)

    raws_concatenated[subject] = mne.concatenate_raws(raw_objects)

raw = mne.concatenate_raws(list(raws_concatenated.values()))

events = mne.find_events(raw)
# Print unique event IDs
unique_event_ids = np.unique(events[:, 2])
print("Unique event IDs:", unique_event_ids)  # [1, 2, 3]


raw = raw.copy().pick(channels) # pick 16 channels out of 64


# ---- Filter design: causal IIR Butterworth band‑pass (8–30 Hz) ----
lowcut = 8.0
highcut = 30.0
frequencies = [lowcut, highcut]
order = 4
fs = 160.0 # sampling frequency
worN = 4096

# Use SOS for actual filtering (stability)
sos = scipy.signal.butter(order, Wn=np.array(frequencies), btype="bandpass", fs=fs, output="sos")

# Apply bandpass filter
signal = raw.get_data()
print(signal.shape)
filtered_signal = scipy.signal.sosfilt(sos, signal)

# Create a new Raw object with the filtered data
filtered_raw = raw.copy()
filtered_raw._data = filtered_signal

event_id = {
    'rest': 1,
    'left_hand': 2,
    'right_hand': 3
}

# TODO: What was meant by the baseline? Should we include it?

epochs = Epochs(
    filtered_raw,
    events,
    event_id=event_id,
    tmin=0.3,
    tmax=3.3, 
    baseline=None, 
    preload=True         
) 
print("Autoreject:")
print(f"Number of epochs before cleaning: {epochs.get_data().shape}")

ar = AutoReject(n_interpolate=[1, 2, 4, 8, 16], random_state=97, verbose=False)
# TODO: do we need to fit-transform on the whole data and not only on training? (NOT on testing though?)
phys_epochs_clean, log = ar.fit_transform(epochs, return_log=True)
thresholds = ar.threshes_       # Dict of channel-level peak-to-peak thresholds

phys_trials = phys_epochs_clean.get_data()
print(f"Number of trials after cleaning: {phys_trials.shape}")

phys_right_trials = phys_epochs_clean["right_hand"].get_data()
phys_left_trials = phys_epochs_clean["left_hand"].get_data()
phys_rest_trials = phys_epochs_clean["rest"].get_data()
print(f"Number of right trials after cleaning: {phys_right_trials.shape}")
print(f"Number of left trials after cleaning: {phys_left_trials.shape}")
print(f"Number of rest trials after cleaning: {phys_rest_trials.shape}")


# ==========================================
# LOADING TARGET-SUBJECT DATA
# ==========================================
print("\n\
------------------------------------------\n\
Loading target-subject data\n\
------------------------------------------")

# For now take only this file since P999 data does not have some channels we are using above
mi_files = [ Path(parent_of_root) / "data" / "sub-P554" / "eeg" / "sub-P554_ses-S002_task-Default_run-001_eeg.xdf"
        ]

subj_raws = []

for f in mi_files:
    fp = f.resolve()
    # print(fp)
    raw, markers, channel_labels = get_raw_offline(fp)
    resampled_raw = raw.resample(160)           # original sampling frequency = 250 
    raw = resampled_raw
    raw = raw.copy().pick(channels)
    subj_raws.append(raw)

subj_raw = mne.concatenate_raws(subj_raws)

# Apply bandpass filter
subj_signal = subj_raw.get_data()
subj_filtered_signal = scipy.signal.sosfilt(sos, subj_signal)

# Create a new Raw object with the filtered data
subj_filtered_raw = subj_raw.copy()
subj_filtered_raw._data = subj_filtered_signal

# Extract the epochs
subj_epochs, _, _ = get_epochs(
    subj_filtered_raw,
    markers,  # ALL markers (of all events)
    tmin=0.3, # Start after the cue period to reduce influence of visual evoked potentials
    tmax=3.3,
)

# Remove bad channels - epochs
print("Autoreject:")
print(f"Number of epochs before cleaning: {subj_epochs.get_data().shape}")
subj_epochs_clean, log = ar.fit_transform(subj_epochs, return_log=True)
subj_trials = subj_epochs_clean.get_data()
print(f"Number of trials after cleaning: {subj_trials.shape}")

subj_right_trials = subj_epochs_clean["ARROW RIGHT"].get_data()
subj_left_trials = subj_epochs_clean["ARROW LEFT"].get_data()
subj_rest_trials = subj_epochs_clean["CIRCLE"].get_data()
print(f"Number of right trials after cleaning: {subj_right_trials.shape}")
print(f"Number of left trials after cleaning: {subj_left_trials.shape}")
print(f"Number of rest trials after cleaning: {subj_rest_trials.shape}")


# Define classes
RIGHT_HAND = 3
LEFT_HAND = 2
REST = 1

phys_trials = np.concatenate((phys_right_trials, phys_left_trials, phys_rest_trials), axis=0)
subj_trials = np.concatenate((subj_right_trials, subj_left_trials, subj_rest_trials), axis=0)
mi_trials = np.concatenate((phys_trials, subj_trials), axis=0)

mi_labels = np.array([RIGHT_HAND] * phys_right_trials.shape[0] + [LEFT_HAND] * phys_left_trials.shape[0] + [REST] * phys_rest_trials.shape[0] +
                     [RIGHT_HAND] * subj_right_trials.shape[0] + [LEFT_HAND] * subj_left_trials.shape[0] + [REST] * subj_rest_trials.shape[0])



print("\n\
------------------------------------------\n\
Concatenated trials\n\
------------------------------------------")
print(f'trials type: {type(mi_trials)}, trials shape: {mi_trials.shape}') # Data in shape (n_trials, n_channels, n_samples)
print(f'lables type: {type(mi_labels)}, labels shape: {mi_labels.shape}')


# ==========================================
# BUILDING WINDOWED TRIALS
# ==========================================

print("\n\
------------------------------------------\n\
Building windowed trials\n\
------------------------------------------")
windowed_trials = []
windowed_labels = []
trial_ids = []

for i, eeg in enumerate(mi_trials):
    windows = extract_overlapping_windows(
        eeg, window_size=160, step_size=32
    )
    windowed_trials.append(windows)
    windowed_labels.extend([mi_labels[i]] * windows.shape[0])
    trial_ids.extend([i] * windows.shape[0])

windowed_trials = np.concatenate(windowed_trials, axis=0)
windowed_labels = np.array(windowed_labels)
trial_ids = np.array(trial_ids)
print(f"Windowed trials shape: {windowed_trials.shape}")
print(f"Windowed labels shape: {windowed_labels.shape}")
print(f"Trial IDs shape: {trial_ids.shape}")

# ==========================================
# SPLITTING THE DATA - TRAIN AND VAL FOR CV
# ==========================================
# Fixed train and validation splits for reproducibility

cv_splits = do_grouped_cv_splits(windowed_trials, windowed_labels, trial_ids, n_splits=5)


# =========================================================
# SPLITTING THE DATA - TRAIN AND TEST FOR FINAL EVALUATION
# =========================================================

data_splits = do_grouped_train_test_split(
    windowed_trials,
    windowed_labels,
    trial_ids,
    random_state=42,
    test_size=0.2,
)

X_train = data_splits["x_train"]
y_train = data_splits["y_train"]
groups_train = data_splits["groups_train"]
X_test = data_splits["x_test"]
y_test = data_splits["y_test"]
groups_test = data_splits["groups_test"]
















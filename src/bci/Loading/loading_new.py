"""
This module handles data loading for the BCI pipeline.

It includes functions to load and preprocess EEG data from the Physionet Motor Imagery dataset
and a target subject's data.

TODO: What does this module need to include?
What else does this module do? Describe its purpose and functionality.
"""

from __future__ import annotations

import warnings
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pyxdf
import json

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
from src.bci.utils.helper import get_raw_xdf_offline, get_epochs


moabb.set_log_level("info")
warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------- #
# TODO: Comments should be written in this way
# --------------------------------------------------------------------------- #

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

def load_physionet_data(subjects: list[int]) -> tuple:
    physionet_root = Path(parent_of_root) / "data" / "datasets" / "physionet"
    raw_dir = physionet_root / "raws"
    event_dir = physionet_root / "events"
    meta_dir = physionet_root / "metadata"

    raw_dir.mkdir(parents=True, exist_ok=True)  
    event_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # If the data is already saved, load it from disk
    if raw_dir.exists() and event_dir.exists():
        # TODO: implement loading from disk
        pass

    # If not, load from Physionet and save to disk
    elif not raw_dir.exists() or not event_dir.exists():
        
        dataset = PhysionetMI(imagined=True, executed=False)
        dataset.feet_runs = []
        all_subjects = dataset.subject_list
        remove = [88, 92, 100]      # despite report, subjects 88, 92 and 100 were sampled with 128 Hz
        target_subjects = [x for x in all_subjects if x not in remove]

        event_dict = dataset.events  
        # {'left_hand': 2, 'right_hand': 3, 'feet': 5, 'hands': 4, 'rest': 1}

        event_id = {
            'rest': event_dict['rest'],
            'left_hand': event_dict['left_hand'],
            'right_hand': event_dict['right_hand'],
        }

        with open(physionet_root / "event_id.json", "w") as f:
            json.dump(event_id, f, indent=2)


        channels = [
                "Fp1","Fp2","F3","Fz","F4","T7","C3","Cz","C4","T8","P3","Pz","P4","PO7","PO8","Oz",
            ]  # Standard 10-20 montage
        
        subject_rows = []

        for subject in target_subjects:
            print(f"Loading subject {subject}")

            raw_dict = dataset.get_data(subjects=[subject])

            raw_list = []
            for session in raw_dict[subject]:
                for run in raw_dict[subject][session]:
                    raw = raw_dict[subject][session][run]
                    duration = raw.n_times / raw.info["sfreq"]
                    raw.set_annotations(
                    mne.Annotations(
                        onset=[0.0],
                        duration=[duration],
                        description=[f"subject_{subject:03d}"]
                                    )
                    )
                    raw_list.append(raw)

            # concatenate runs of THIS subject only
            raw = mne.concatenate_raws(raw_list)
            raw.pick(channels)

            events = mne.find_events(raw)

            # save raw
            raw_path = raw_dir / f"subj_{subject:03d}_raw.fif"
            raw.save(raw_path, overwrite=True)

            # save events
            np.save(event_dir / f"subj_{subject:03d}_events.npy", events)

            # register metadata
            subject_rows.append({
                "subject_id": subject,
                "n_channels": len(raw.ch_names),
                "n_samples": raw.n_times,
                "sfreq": raw.info["sfreq"],
                "raw_file": raw_path.name,
                "events_file": f"subj_{subject:03d}_events.npy",
            })

            # free memory explicitly
            del raw, raw_list, raw_dict

        df = pd.DataFrame(subject_rows)
        df.to_csv(meta_dir / "subjects.csv", index=False)

    # Load the saved data from disk
    raws_concatenated = {}
    for subject in subjects:
        raw = mne.read_raw_fif(f"subj_{subject:03d}_raw.fif", preload=True)
        raws_concatenated[subject] = raw

        events = np.load(f"subj_{subject:03d}_events.npy")


    return raws, events, event_id, subject_sample_ids





physionet_folder = Path(parent_of_root) / "data" / "datasets" / "physionet"
physionet_folder.mkdir(parents=True, exist_ok=True)

for subject in subjects:
    print(f"Processing subject {subject}")

    # Load ONE subject only
    raw_dict = dataset.get_data(subjects=[subject])

    raw_objects = []
    for session in raw_dict[subject]:
        for run in raw_dict[subject][session]:
            raw_objects.append(raw_dict[subject][session][run])

    # Concatenate runs of THIS subject only
    raw = mne.concatenate_raws(raw_objects)

    # Pick channels early (saves memory)
    raw.pick(channels)

    # Save per-subject raw
    raw.save(
        physionet_folder / f"physionet_subj_{subject:03d}_raw.fif",
        overwrite=True
    )

    # Explicit cleanup
    del raw, raw_objects, raw_dict









raw_dict = dataset.get_data(subjects=subjects)
''' E.g. for subject 2:
{2: 
    {'0': -> we want to index these amoung subjects (see whther the runs are very near each other in time)
        {'0': <RawEDF | S002R04.edf, 65 x 19680 (123.0 s), ~9.8 MiB, data loaded>, 
         '1': <RawEDF | S002R08.edf, 65 x 19680 (123.0 s), ~9.8 MiB, data loaded>, 
         '2': <RawEDF | S002R12.edf, 65 x 19680 (123.0 s), ~9.8 MiB, data loaded>
        }
    }
}
'''

raws_concatenated = {}
subject_sample_ids = []  # Keep track which sample belongs to which subject

for subject in subjects:
    raw_objects = []
    for session in raw_dict[subject]:   # subject level
        for run in raw_dict[subject][str(session)]:  # run level
            raw = raw_dict[subject][str(session)][str(run)]
            raw_objects.append(raw)
    subject_raw = mne.concatenate_raws(raw_objects)
    raws_concatenated[subject] = subject_raw

    # For each sample in this subject's raw, assign the subject id
    subject_sample_ids.extend([subject] * subject_raw.n_times)

raw = mne.concatenate_raws(list(raws_concatenated.values()))
raw = raw.copy().pick(channels) # pick 16 channels out of 64

events = mne.find_events(raw)
# Print unique event IDs
unique_event_ids = np.unique(events[:, 2])
print("Unique event IDs:", unique_event_ids)  # [1, 2, 3]

# TODO: save raws, save events, save event_ids, save subject_sample_ids
physionet_folder = Path(parent_of_root) / "data" / "datasets"/ "physionet"

# np.save(physionet_folder / "subject_sample_ids.npy", subject_sample_ids)

raw.save(physionet_folder / "physionet_raw.fif", overwrite=False) # avoid overwriting existing file
np.savez(physionet_folder / "physionet_events_subjects_channels.npz",
         events=events,
         event_id=event_id,
         subject_sample_ids=subject_sample_ids,
         channels=channels
         )

# raw = mne.io.read_raw_fif("physionet_raw.fif", preload=True)
# data = np.load("physionet_events_subjects_sessions.npz", allow_pickle=True)
# events = data["events"]
# subjects = data["subjects"]
# subject_session_idx = data["subject_session_idx"]

# TODO: What was meant by the baseline? Should we include it?

i 

# ==========================================
# LOADING TARGET-SUBJECT DATA
# ==========================================
print("\n\
------------------------------------------\n\
Loading target-subject data\n\
------------------------------------------")

folder = Path(parent_of_root) / "eeg" / "data" / "sub-P999" / "eeg"
print("Checking xdf files in folder:", folder)
print("\n")

xdf_files: list[Path] = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".xdf"
    ]
print(f"Found {len(xdf_files)} xdf files in folder {folder}.\n")
print(f"the files are: {[x.name for x in xdf_files]}\n")
    
for file in xdf_files:
    print(f"Processing file: {file.name}")
    raw, markers, channel_labels = get_raw_xdf_offline(file)
    if raw is None:
        print(f"  Skipping file {file.name} due to incompatible channel information.\n")
        continue
    print(f"  Number of channels: {len(channel_labels)}, channels: {channel_labels}\n")

# For now take only this file since P999 data does not have some channels we are using above
mi_files = [ Path(parent_of_root) / "data" / "sub-P554" / "eeg" / "sub-P554_ses-S002_task-Default_run-001_eeg.xdf"
        ]

subj_raws = []

for f in mi_files:
    fp = f.resolve()
    # print(fp)
    raw, markers, channel_labels = get_raw_xdf_offline(fp)
    print(markers)
    resampled_raw = raw.resample(160)           # original sampling frequency = 250 
    raw = resampled_raw
    raw = raw.copy().pick(channels)
    subj_raws.append(raw)

subj_raw = mne.concatenate_raws(subj_raws)




# Define classes
RIGHT_HAND = 3
LEFT_HAND = 2
REST = 1


# ==========================================
# SPLITTING THE DATA - TRAIN AND VAL FOR CV
# ==========================================
# Fixed train and validation splits for reproducibility

# cv_splits = do_grouped_cv_splits(windowed_trials, windowed_labels, trial_ids, n_splits=5)


# =========================================================
# SPLITTING THE DATA - TRAIN AND TEST FOR FINAL EVALUATION
# =========================================================

# data_splits = do_grouped_train_test_split(
#     windowed_trials,
#     windowed_labels,
#     trial_ids,
#     random_state=42,
#     test_size=0.2,
# )

# X_train = data_splits["x_train"]
# y_train = data_splits["y_train"]
# groups_train = data_splits["groups_train"]
# X_test = data_splits["x_test"]
# y_test = data_splits["y_test"]
# groups_test = data_splits["groups_test"]
















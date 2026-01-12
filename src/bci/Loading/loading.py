"""
loading Module: This module handles data loading for the BCI pipeline.

This module handles:
1. Loading Physionet Motor Imagery Data
2. Loading Target Subject Data

Usage:
    from src.bci import loading
    
    # Load Physionet data for subjects 1 to 5
    physionet_raws, 
    physionet_events, 
    physionet_event_id, 
    physionet_subject_ids = loading.load_physionet_data(
        subjects=list(range(1, 6)),
        parent_of_root="/path/to/parent/of/root"
    )
    
    # Load Target Subject data (Subject 110)
    target_raws,
    target_events,
    target_event_id,
    target_subject_ids = loading.load_target_subject_data(
        parent_of_root="/path/to/parent/of/root",
        source_folder="/path/to/source/xdf/files",
        task_type="arrow",  # or "dino" or "all"
        limit=5  # Optional limit on number of files to load
    )
    
"""

from __future__ import annotations

import warnings
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import json

import mne
from mne.io import concatenate_raws

import moabb

# Add the root directory to the Python path
root_dir = str(Path(__file__).resolve().parents[3])  # BCI-Challange directory
parent_of_root = str(Path(__file__).resolve().parents[4])  # one level above

sys.path.append(root_dir)
sys.path.append(parent_of_root)

sys.path.append("..")

moabb.set_log_level("info")
warnings.filterwarnings("ignore")



def get_raw_xdf_offline(
    trial: Path, marker_durations: list[float] | None = None
) -> tuple[mne.io.RawArray, list, list[str]]:
    """
    Function to load the raw data from the trial and return the raw data object, 
    the markers and the channel labels.
    Based on the implementation from https://gitlab.lrz.de/students2/baseline-bci. 
    
    Parameters
    ----------
    trial : Path
        Path to the trial file.
    marker_durations : list, optional
        List of marker durations. The default is [3, 1, 3].

    Returns
    -------
    raw_data : mne.io.RawArray
        Raw data object.
    markers : list
        List of markers.
    channel_labels : list[str]
        List of channel labels.

    Notes
    -----
    The current implementation processes two types of recordings:
    1. Recordings with no channel information (assigns predefined channel labels)
    2. Recordings where channel information exactly matches predefined channel labels
    
    Number of channels should be 16, the order and naming of channels should match the list 
    of channel_labels below.
    Function may need to be adapted for other recordings.
    """
    import pyxdf

    # Define channel labels
    expected_channel_labels = [
        "Fp1","Fp2","F3","Fz","F4","T7","C3","Cz","C4","T8","P3","Pz","P4","PO7","PO8","Oz",
    ]  # Manually set for g.Nautilus, standard 10-20 montage

    # Load the data
    if marker_durations is None:
        marker_durations = [3, 1, 3]

    streams, header = pyxdf.load_xdf(trial, verbose=False)

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

    # print(streams[eeg_channel]["info"])

    # Validate channel information:
    if streams[eeg_channel]["info"]["desc"] != [None]:  
        print("Channel info found in the recording.")
        # Channel info is available - check if it matches expected labels
        channel_dict = streams[eeg_channel]["info"]["desc"][0]["channels"][0]["channel"]
        eeg_channels = [channel["label"][0] for channel in channel_dict]
        
        # Check if channels match expected labels
        if eeg_channels != expected_channel_labels:
            # Channel info exists but doesn't match - filter out this recording
            return None, None, None
    else:  
        print("No channel info found in the recording.")
        # No channel info available - use predefined channel labels
        channel_labels = expected_channel_labels


    # Create montage object - this is needed for the raw data object (Layout of the electrodes)
    montage = mne.channels.make_standard_montage("standard_1020")
    ts = streams[eeg_channel]["time_series"][-1]
    print(
        f"EEG data timestamps range from {ts[0]/250.0} to {ts[-1]/250.0} (total {ts[-1]/250.0-ts[0]/250.0:.2f} seconds)"
    )

    # Get EEG data - https://mne.tools/dev/auto_examples/io/read_xdf.html#ex-read-xdf
    data = streams[eeg_channel]["time_series"].T * 1e-6  # scaling the data to volts
    print(len(data), data.shape)

    # # exclude last channel if it is impedance
    # if channel_labels[-1].lower() in ["ts", "impedances", "keyboard"]:
    #     data = data[:-1, :]
    #     channel_labels = channel_labels[:-1]

    # Verify we have the correct number of channels
    if data.shape[0] != len(channel_labels):
        print(f"Data has {data.shape[0]} channels, but expected {len(channel_labels)} channels")
        return None, None, None

    # Get sampling frequency and create info object
    sfreq = float(streams[eeg_channel]["info"]["nominal_srate"][0])
    info = mne.create_info(channel_labels, sfreq, ch_types="eeg")

    # Create raw object and set the montage
    raw_data = mne.io.RawArray(data, info, verbose=False)
    raw_data.set_montage(montage)

    # In the case where the offline collected data (calibration from online) does not have any markers
    if event_channel is None:
        return raw_data, markers, channel_labels

    # get naming of markers and convert to numpy array
    marker = np.array(streams[event_channel]["time_series"]).squeeze()

    # get time stamps of markers
    time_marker = np.array(streams[event_channel]["time_stamps"]).squeeze()

    # get time stamps of data
    time_data = np.array(streams[eeg_channel]["time_stamps"])

    # get relative time of markers
    real_time_marker = (time_marker - time_data[0]).astype(float)

    # Create array of durations for each individual marker
    duration_list = np.zeros(len(real_time_marker))
    for i, _duration in enumerate(duration_list):
        duration_list[i] = marker_durations[i % len(marker_durations)]

    # Annotate the raw data with the markers (So that we know what events are happening at what time in the data)
    annotations = mne.Annotations(
        onset=real_time_marker, duration=duration_list, description=marker
    )
    raw_data.set_annotations(annotations)

    # Save the unique markers for later use
    markers = list(set(marker))
    markers.sort()

    return raw_data, markers, channel_labels



def load_physionet_data(subjects: list[int], parent_of_root: str) -> tuple:
    '''Load Physionet Motor Imagery data for specified subjects.
    
    This function checks if the data is already saved on disk. If not, it downloads
    the data from Physionet, preprocesses it (attaching subject IDs), and saves it
    to disk for future use.

    Parameters
    ----------
    subjects : list[int]
        List of subject IDs to load. IDs are integers.
    parent_of_root : str
        Path to the parent directory of the root dataset folder.

    Returns
    -------
    tuple
        A tuple containing:
        - loaded_raws: List of MNE Raw objects for each subject.
        - loaded_events: List of event arrays for each subject.
        - event_id: Dictionary mapping event labels to event IDs.
        - subject_ids_out: List of subject IDs corresponding to the loaded data.
    '''

    physionet_root = Path(parent_of_root) / "data" / "datasets" / "physionet"
    raw_dir = physionet_root / "raws"
    event_dir = physionet_root / "events"
    meta_dir = physionet_root / "metadata"
    event_id_path = physionet_root / "event_id.json"
    subjects_csv = meta_dir / "subjects.csv"

    for d in [raw_dir, event_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. DATA GENERATION / SAVING TO DISK
    if not subjects_csv.exists():
        from moabb.datasets import PhysionetMI
        dataset = PhysionetMI(imagined=True, executed=False)
        dataset.feet_runs = []
        
        remove = [88, 92, 100]
        target_subjects = [x for x in dataset.subject_list if x not in remove]

        event_dict = dataset.events  
        event_id = {'rest': event_dict['rest'], 
                    'left_hand': event_dict['left_hand'], 
                    'right_hand': event_dict['right_hand']}
        
        # Save event_id mapping
        with open(event_id_path, "w") as f:
            json.dump(event_id, f, indent=2)

        channels = ["Fp1",
                    "Fp2",
                    "F3",
                    "Fz",
                    "F4",
                    "T7",
                    "C3",
                    "Cz",
                    "C4",
                    "T8",
                    "P3",
                    "Pz",
                    "P4",
                    "PO7",
                    "PO8",
                    "Oz"]   # standard 10-20 montage
        subject_rows = []

        for sub_id in target_subjects:
            raw_dict = dataset.get_data(subjects=[sub_id])
            raw_list = []
            for session in raw_dict[sub_id]:
                for run in raw_dict[sub_id][session]:
                    raw_list.append(raw_dict[sub_id][session][run])

            raw_sub = mne.concatenate_raws(raw_list)
            raw_sub.pick(channels)
            
            # --- ATTACH SUBJECT ID TO RAW ---
            # Use the 'subject_info' dictionary which is the standard MNE way
            raw_sub.info['subject_info'] = {'id': sub_id}
            
            raw_path = raw_dir / f"subj_{sub_id:03d}_raw.fif"
            events_path = event_dir / f"subj_{sub_id:03d}_events.npy"
            
            events_sub, _ = mne.events_from_annotations(raw_sub, event_id=event_id)
            np.save(events_path, events_sub)
            raw_sub.save(raw_path, overwrite=True)
            # np.save(events_path, mne.find_events(raw_sub))
            
            # Record metadata to CSV
            subject_rows.append({"subject_id": sub_id, "raw_file": raw_path.name, "events_file": events_path.name})

        pd.DataFrame(subject_rows).to_csv(subjects_csv, index=False)
    
    # 2. LOAD DATA FROM DISK
    with open(event_id_path, "r") as f:
        event_id = json.load(f)

    loaded_raws = []
    loaded_events = []
    subject_ids_out = []

    for sub_id in subjects:
        if sub_id in [88, 92, 100]:
            continue  # Skip subjects with known issues

        r_path = raw_dir / f"subj_{sub_id:03d}_raw.fif"
        e_path = event_dir / f"subj_{sub_id:03d}_events.npy"
        
        raw = mne.io.read_raw_fif(r_path, preload=True)
        evs = np.load(e_path)
        
        # Ensure the ID is present in the info after loading
        raw.info['subject_info'] = {'id': sub_id}
        
        loaded_raws.append(raw)
        loaded_events.append(evs)
        subject_ids_out.append(sub_id)

    return loaded_raws, loaded_events, event_id, subject_ids_out



# TODO: Function in progress
def load_target_subject_data(
    parent_of_root: str, 
    source_folder: str = None, 
    task_type: str = "all", 
    limit: int = None
) -> tuple:
    """
    Loads target data (Subject 110). 
    1. Checks 'data/datasets/target' for existing .fif files.
    2. If empty, processes XDF from source_folder.
    3. Infers and saves event_id mapping.
    """
    target_dir = Path(parent_of_root) / "data" / "datasets" / "target"
    raw_save_dir = target_dir / "raws"
    event_save_dir = target_dir / "events"
    event_id_path = target_dir / "event_id.json"
    
    raw_save_dir.mkdir(parents=True, exist_ok=True)
    event_save_dir.mkdir(parents=True, exist_ok=True)

    # --- STEP 1: CHECK TARGET FOLDER ---
    existing_fif = sorted(list(raw_save_dir.glob("*.fif")))
    
    # Filter by task_type if existing files are found
    if task_type == "dino":
        existing_fif = [f for f in existing_fif if "dino" in f.name.lower()]
    elif task_type == "arrow":
        existing_fif = [f for f in existing_fif if "dino" not in f.name.lower()]

    if len(existing_fif) > 0:
        print(f"Found {len(existing_fif)} processed files in target folder. Loading...")
        if limit: existing_fif = existing_fif[:limit]   # load files only up to limit
        
        loaded_raws = []
        loaded_events = []
        for fif_p in existing_fif:
            raw = mne.io.read_raw_fif(fif_p, preload=True)
            # Find corresponding event file
            npy_p = event_save_dir / f"{fif_p.stem.replace('_raw', '')}_events.npy"
            evs = np.load(npy_p)
            # print(evs)
            loaded_raws.append(raw)
            loaded_events.append(evs)
        
        with open(event_id_path, 'r') as f:
            event_id = json.load(f)
            # print(event_id)
            
        return loaded_raws, loaded_events, event_id, [110] * len(loaded_raws)

    # --- STEP 2: IF TARGET EMPTY, CHECK SOURCE ---
    if source_folder is None:
        raise ValueError("Target folder is empty and no source_folder was provided to process raw XDF files.")

    source_path = Path(source_folder)
    all_xdf = sorted([p for p in source_path.glob("*.xdf")])
    
    if task_type == "dino":
        selected_files = [f for f in all_xdf if "dino" in f.name.lower()]
    elif task_type == "arrow":
        selected_files = [f for f in all_xdf if "dino" not in f.name.lower()]
    else:
        selected_files = all_xdf

    if not selected_files:
        raise FileNotFoundError(f"No XDF files matching category '{task_type}' found in {source_folder}")

    if limit: selected_files = selected_files[:limit]

    loaded_raws, loaded_events, selected_event_id = [], [], {}
    
    channels = ["Fp1",
                "Fp2",
                "F3",
                "Fz",
                "F4",
                "T7",
                "C3",
                "Cz",
                "C4",
                "T8",
                "P3",
                "Pz",
                "P4",
                "PO7",
                "PO8",
                "Oz"]   # standard 10-20 montage

    # --- STEP 3: PROCESS XDF AND INFER EVENT_ID ---
    for file_path in selected_files:
        print(f"Processing XDF: {file_path.name}")
        raw, markers, channel_labels = get_raw_xdf_offline(file_path)
        
        if raw is None: continue

        raw.resample(160)
        raw.pick(channels)

        # Rename annotations to standard labels (not sure if we need it though)
        # mapping = {
        #     "CIRCLE": "rest",
        #     "ARROW LEFT": "left_hand",
        #     "ARROW RIGHT": "right_hand"
        # }

        # # Apply the rename directly to the raw object
        # raw.annotations.rename(mapping)

        # Infer event_id: Get unique descriptions from annotations
        # This builds a dictionary {description: integer} automatically
        current_events, current_event_id = mne.events_from_annotations(raw)
        # print(current_events)
        print("Inferred event IDs:", current_event_id)
        
        event_id = {'CIRCLE': current_event_id['CIRCLE'], 
                    'ARROW LEFT': current_event_id['ARROW LEFT'], 
                    'ARROW RIGHT': current_event_id['ARROW RIGHT']}

        # NOTE: Might need to ajust event id numbers for further classfication
        # For now: {'rest': 29, 'left_hand': 24, 'right_hand': 26}

        # NOTE: For Dino, we have different events, so we need to handle them separately
        ''' Events IDs for Dino files
        {np.str_(''): 1, np.str_('ARROW LEFT ONSET'): 2, 
         np.str_('ARROW RIGHT ONSET'): 3, 
         np.str_('CIRCLE ONSET'): 4, 
         np.str_('JUMP'): 5, 
         np.str_('JUMP FAIL'): 6}
         '''
        # print(current_event_id)
        selected_event_id.update(event_id)

        # Save files
        base_name = file_path.stem  # filename without extension
        raw.info['subject_info'] = {'id': 110}
        raw.save(raw_save_dir / f"{base_name}_raw.fif", overwrite=True)
        np.save(event_save_dir / f"{base_name}_events.npy", current_events)

        loaded_raws.append(raw)
        loaded_events.append(current_events)

    # Save the inferred event_id for future runs
    with open(event_id_path, "w") as f:
        json.dump(selected_event_id, f, indent=2)

    return loaded_raws, loaded_events, selected_event_id, [110] * len(loaded_raws)



# ==========================================
# sub_raws, sub_events, sub_event_id, sub_ids = load_target_subject_data(
#     parent_of_root=parent_of_root,
#     source_folder=str(Path(parent_of_root) / "data" / "eeg" / "sub-P999" / "eeg"),
#     task_type="arrow",
#     limit=None)

# print(sub_event_id)
# print(f"Loaded {len(sub_raws)} raws from target subject data.")
# print("Subject IDs:", sub_ids)
# ==========================================


'''
# NOTE: The following is an example of how to use the loaded data for you guys as a reference
# on how the data could be processed further.
# TODO: Needs to be removed from the final version.
# ==========================================
# Small example to load data
# ==========================================

subjects_to_load = list(range(1, 2))

physionet_loaded_raws, physionet_loaded_events, physionet_event_id, physionet_subject_ids = load_physionet_data(
    subjects=subjects_to_load,
    parent_of_root=parent_of_root
)

print(f"Loaded {len(physionet_loaded_raws)} subjects from Physionet.")
print("Subject IDs:", physionet_subject_ids)

# ==========================================
# SAMPLE preprocessing: Filtering, Epoching, Metadata attachment with the function above
# ==========================================
# Assume 'loaded_raws' and 'loaded_events' are the lists from the physionet loading function
from mne import Epochs, pick_types
all_epochs_list = []

for raw, events, sub_id in zip(physionet_loaded_raws, physionet_loaded_events, physionet_subject_ids):
    # A. Filtering (Subject identity is safe inside the 'raw' object)
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    
    # B. Create Epochs
    epochs = mne.Epochs(raw, events, event_id=physionet_event_id, tmin=-0.2, tmax=4.0, preload=True)
    
    # C. ATTACH METADATA
    # We create a dataframe where each row corresponds to an epoch (trial)
    metadata = pd.DataFrame({
        'subject_id': [sub_id] * len(epochs),
        'condition': epochs.events[:, 2]  # Optional: track class labels here too
    })
    epochs.metadata = metadata
    
    all_epochs_list.append(epochs)

# D. Concatenate all subjects into one object
combined_epochs = mne.concatenate_epochs(all_epochs_list)

# ==========================================
# SAMPLE usage with Scikit-Learn GroupKFold
# ==========================================
# Prepare for Scikit-Learn
X = combined_epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
y = combined_epochs.events[:, 2] # The labels (e.g., 1, 2, 3)
groups = combined_epochs.metadata['subject_id'].values  # Subject IDs for grouping

# Cross-Validation
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)

for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train subjects: {np.unique(groups[train_idx])}, Test subjects: {np.unique(groups[test_idx])}")

'''













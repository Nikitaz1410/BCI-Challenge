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
        root="/path/to/root_dir"
    )

    # Load Target Subject data (Subject 110)
    target_raws,
    target_events,
    target_event_id,
    target_subject_ids = loading.load_target_subject_data(
        root="<BCI-Challange root directory>",
        source_folder="/root/data/eeg/sub-P999/eeg",
    )

TODO: The function should load any data (P554 with 16 non-standard channels),
P999 with 16 standard channels, P999 with 23 standard channels (from which 13 are matching anything else).
Physionet loading function should load all 64 channels.
"""

from __future__ import annotations

from os import PathLike
import warnings
import sys
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json

import mne
from mne.io import concatenate_raws

import moabb
from moabb.datasets import PhysionetMI, Cho2017

from bci.utils.bci_config import EEGConfig

# Add the root directory to the Python path
root_dir = str(Path(__file__).resolve().parents[3])  # BCI-Challange directory
sys.path.append(root_dir)

# sys.path.append("..")

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _get_raw_xdf_offline(
    trial: Path, marker_durations: list[float] | None = None, 
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
    TODO: Now we have more data, description needs to be revised
    The current implementation processes following types of recordings:
    1. Recordings with 16 channels and no channel information -> assign 16 channel labels from standard 1020 montage
    2. Recordings with 16 channels, and channel information exactly matches standard channel labels -> use existing channel labels
    3. Recordings with 17 channels -> P554 recordings with custom 16 channel labels + keyboard -> reorder channels to match standard
    4. Recordings with 24 channels -> discard the recording (different markers)

    P554 recordings of types "blinking", "jaw_clenching", "Music" are discarded.
    """
    import pyxdf

    print("\n")
    print(("=" * 30) + f" Processing file: {trial.name} " + ("=" * 30))
    print(f"Loading trial from: {trial}")

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
    standard_channels = [
        "Fp1",
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
        "Oz",
    ]

    target_codes = ["S001", "S101", "S102", "S103", "S104"]

    # Iterate through streams to find desired indices, extract EEG stream
    for i, stream in enumerate(streams):
        name = stream["info"]["name"][0]  # name of the stream
        if name == "EEG - Impedances":
            print("EEG - Impedances stream found, skipping...")
            continue
            _index_impedance = i  # impedance stream
        elif name == "EEG" or name == "EEG-GTEC":
            eeg_channel = i  # eeg stream
            print("EEG stream found.")
        elif name == "pupil_capture_pupillometry_only":
            print("Pupil capture (pupillometry only) stream found.")
            continue
            pupil_channel = i  # pupil stream
        elif name == "pupil_capture":
            print("Pupil capture stream found.")
            continue
            pupil_capture_channel = i  # pupil stream
        elif name == "pupil_capture_fixations":
            print("Pupil capture fixations stream found.")
            continue
            pupil_capture_fixations_channel = i
        else:
            print(f"Unknown stream found: {name}")
            event_channel = i  # markers stream


    # Validate channel information:
    if streams[eeg_channel]["info"]["desc"] != [None]:
        print("Channel info found in the recording.")
        # Channel info is available - check if it matches expected labels
        channel_dict = streams[eeg_channel]["info"]["desc"][0]["channels"][0]["channel"]
        eeg_channels = [channel["label"][0] for channel in channel_dict]

        print(f"Channels in recording, total {len(eeg_channels)}:", eeg_channels)
        channel_labels = eeg_channels

        # Check if channels match expected labels
        if eeg_channels != standard_channels:
            # Channel info exists but doesn't match - filter out this recording
            print("Channels do not match standard 10-20 montage.")
            channel_labels = eeg_channels

        if len(eeg_channels) == 24:
            print(
                "Recording with 24 channels detected. Different markers, so discarding the recording..."
            )
            return None, None, None

    else:
        print("No channel info found in the recording.")
        print(
            "Number channels detected:", streams[eeg_channel]["info"]["channel_count"]
        )
        # No channel info available - use predefined channel labels

        if streams[eeg_channel]["info"]["channel_count"][0] == "16":
            # Cut the last channel
            print(
                "Number of channels is 16. Assigning 16 first channel labels from standard_1020 montage..."
            )
            channel_labels = standard_channels

        if "P999" in trial.name and any(code in trial.name for code in target_codes):
            print(
                "Switched channels Cz and Fp2 detected. Assigning custom 16 channel labels + keyboard..."
            )
            custom_channels_p999 = [
                "Fp1",
                "Cz",
                "F3",
                "Fz",
                "F4",
                "T7",
                "C3",
                "Fp2",
                "C4",
                "T8",
                "P3",
                "Pz",
                "P4",
                "PO7",
                "PO8",
                "Oz",
                "Keyboard",
            ]
            channel_labels = custom_channels_p999  # Switched Fp2 and Cz for P999 with 16 channels + keyboard

        if "P124" in trial.name:
            print(
                "P124 recording detected. Recording empty, discarding..."
            )
            return None, None, None
        
        if "P554" in trial.name:
            print(
                "P554 recording detected. Assigning custom 16 channel labels + keyboard..."
            )
            custom_channels_p554 = [
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
                "Keyboard",
            ]
            channel_labels = (
                custom_channels_p554  # Custom channel labels for P554 with 16 channels
            )

        if (
            ("blinking" in trial.name)
            or ("jaw_clenching" in trial.name)
            or ("Music" in trial.name)
        ):
            print(
                "Blinking or jaw clenching or music detected. Discarding the recording..."
            )
            return None, None, None

    # Create montage object - this is needed for the raw data object (Layout of the electrodes)
    montage = mne.channels.make_standard_montage("standard_1020")
    ts = streams[eeg_channel]["time_series"][-1]
    print(
        f"EEG data timestamps range from {ts[0]/250.0} to {ts[-1]/250.0} (total {ts[-1]/250.0-ts[0]/250.0:.2f} seconds)"
    )

    # Get EEG data - https://mne.tools/dev/auto_examples/io/read_xdf.html#ex-read-xdf
    data = streams[eeg_channel]["time_series"].T * 1e-6  # scaling the data to volts

    if channel_labels[-1].lower() in ["ts", "impedances", "keyboard"]:
        print("Discarding the last channel (non-EEG channel).")
        data = data[:-1, :]
        channel_labels = channel_labels[:-1]

    # Verify we have the correct number of channels
    if data.shape[0] != len(channel_labels):
        print(
            f"Data has {data.shape[0]} channels, but expected {len(channel_labels)} channels."
        )
        print("Discarding this recording from further processing...")
        return None, None, None

    # Get sampling frequency and create info object
    sfreq = float(streams[eeg_channel]["info"]["nominal_srate"][0])
    info = mne.create_info(channel_labels, sfreq, ch_types="eeg")

    print("channelds from info:", info["ch_names"])

    # Create raw object and set the montage
    raw_data = mne.io.RawArray(data, info, verbose=False)
    raw_data.set_montage(montage)

    if "P554" in trial.name or ("P999" in trial.name and any(code in trial.name for code in target_codes)):
        print("Reordering channels for standard 10-20 montage...")
        raw_data.reorder_channels(standard_channels)
        channel_labels = raw_data.ch_names

    print(
        f"Final channel labels used ({len(raw_data.ch_names)} channels):",
        raw_data.ch_names,
    )

    # raw_data.filter(l_freq=1.0, h_freq=40.0)
    # raw_data.plot()
    # fig = raw_data.plot(duration=30, n_channels=16, show=False, block=False)

    # print(f"Max value: {np.max(raw_data.get_data())}")
    # print(f"Min value: {np.min(raw_data.get_data())}")
    # print(f"Mean value: {np.mean(raw_data.get_data())}")

    # # Save it
    # save_dir = Path(root_dir)
    # save_path = save_dir / "data"/ f"inspect_{trial.name}.png"
    # fig.savefig(save_path, dpi=200)
    # plt.close(fig)

    # In the case where the offline collected data (calibration from online) does not have any markers
    if event_channel is None:
        print("No event channel found in the recording.")
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

    if len(raw_data.annotations) == 0:
        print("No annotations found after adding. Discarding the recording...")
        return None, None, None

    # Save the unique markers for later use
    markers = list(set(marker))
    markers.sort()

    return raw_data, markers, channel_labels


def _standardize_and_map(raw, target_event_id):
    """
    Standardizes markers with automatic detection of marker type.
    
    Handles files with:
    - Only "<EVENT>" markers (e.g., "ARROW LEFT")
    - Only "<EVENT> ONSET" markers (e.g., "ARROW LEFT ONSET")
    - Both types (prefers "<EVENT>" over "<EVENT> ONSET")
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw object with annotations
    target_event_id : dict
        Final event ID mapping, e.g., {"rest": 0, "left_hand": 1, "right_hand": 2}
    mode : str NOTE
        "general" or "dino" (currently unused, auto-detected)
    
    Returns
    -------
    events : ndarray
        Events array with standardized event codes
    
    Raises
    ------
    ValueError
        If no matching annotations found or no events extracted
    """

    # 1. Define mappings specific to each recording type
    rename_map_base = {
        "ARROW LEFT": "left_hand",
        "ARROW RIGHT": "right_hand",
        "CIRCLE": "rest",
    }
    
    rename_map_onset = {
        "ARROW LEFT ONSET": "left_hand",
        "ARROW RIGHT ONSET": "right_hand",
        "CIRCLE ONSET": "rest",
    }

    # 2. Apply the rename to the raw object
    existing_descriptions = set(raw.annotations.description)
    # print(f"Existing annotations before standardization: {existing_descriptions}")

    existing_descriptions = {str(desc) for desc in existing_descriptions}

    base_markers_present = any(k in existing_descriptions for k in rename_map_base.keys())
    onset_markers_present = any(k in existing_descriptions for k in rename_map_onset.keys())
    
    print(f"  Base markers ('<EVENT>') present: {base_markers_present}")
    print(f"  Onset markers ('<EVENT> ONSET') present: {onset_markers_present}")
    
    # 4. Choose which mapping to use (priority: base > onset)
    if base_markers_present and onset_markers_present:
        print(f"  Both marker types found! Using base markers '<EVENT>' (higher priority)")
        rename_map = rename_map_base
    elif base_markers_present:
        print(f"  Using base markers '<EVENT>'")
        rename_map = rename_map_base
    elif onset_markers_present:
        print(f"  Using onset markers '<EVENT> ONSET'")
        rename_map = rename_map_onset
    else:
        raise ValueError(
            f"No matching markers found!\n"
            f"  Expected either:\n"
            f"    - Base: {list(rename_map_base.keys())}\n"
            f"    - Onset: {list(rename_map_onset.keys())}\n"
            f"  Found: {existing_descriptions}"
        )

    actual_map = {k: v for k, v in rename_map.items() if k in existing_descriptions}
    print(f"  Renaming {len(actual_map)} types: {actual_map}")
    raw.annotations.rename(actual_map)
    
    # print(f"Annotations after standardization: {set(raw.annotations.description)}")

    # 3. Force extraction of ONLY the standardized keys
    try:
        events, _ = mne.events_from_annotations(raw, event_id=target_event_id)
    except ValueError as e:
        raise ValueError(
            f"Failed to extract events.\n"
            f"  target_event_id: {target_event_id}\n"
            f"  Annotations after rename: {raw.annotations.description}\n"
            f"  Original error: {e}"
        )
    
    if len(events) == 0:
        raise ValueError(
            f"No valid events extracted.\n"
            f"  target_event_id: {target_event_id}\n"
            f"  Annotations after rename: {raw.annotations.description}"
        )

    return events


def load_physionet_data(subjects: list[int], root: Path, channels: list[str]) -> tuple:
    """Load Physionet Motor Imagery data for specified subjects.

    This function checks if the data is already saved on disk. If not, it downloads
    the data from Physionet, preprocesses it (attaching subject IDs), and saves it
    to disk for future use.

    Parameters
    ----------
    subjects : list[int]
        List of subject IDs to load. IDs are integers.
    root : str
        Path to the root directory of the dataset.
    channels : list[str]
        List of channel names to load from the raw data.

    Returns
    -------
    tuple
        A tuple containing:
        - loaded_raws: List of MNE Raw objects for each subject.
        - loaded_events: List of event arrays for each subject.
        - event_id: Dictionary mapping event labels to event IDs.
        - subject_ids_out: List of subject IDs corresponding to the loaded data.
        - raw_filenames: List of raw filenames corresponding to the loaded data.
    """

    physionet_root = root / "data" / "datasets" / "physionet"
    raw_dir = physionet_root / "raws"
    event_dir = physionet_root / "events"
    meta_dir = physionet_root / "metadata"
    event_id_path = physionet_root / "event_id.json"
    subjects_csv = meta_dir / "subjects.csv"

    for d in [raw_dir, event_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. DATA GENERATION / SAVING TO DISK
    if not subjects_csv.exists():
        dataset = PhysionetMI(imagined=True, executed=False)
        dataset.feet_runs = []

        remove = [88, 92, 100]
        target_subjects = [x for x in dataset.subject_list if x not in remove]

        # event_dict = dataset.events (other integers: 1, 2, 3)
        event_id = {
            "rest": 0,
            "left_hand": 1,
            "right_hand": 2,
        }

        # Save event_id mapping
        with open(event_id_path, "w") as f:
            json.dump(event_id, f, indent=2)

        print("Processing and saving Physionet data to disk...")

        subject_rows = []

        for sub_id in target_subjects:
            raw_dict = dataset.get_data(subjects=[sub_id])
            raw_list = []
            for session in raw_dict[sub_id]:
                for run in raw_dict[sub_id][session]:
                    raw_list.append(raw_dict[sub_id][session][run])

            raw_sub = mne.concatenate_raws(raw_list)

            if channels is not None:
                raw_sub.pick(channels)
            else:
                print("Keeping all 64 channels for now...")

            # --- ATTACH SUBJECT ID TO RAW ---
            # Use the 'subject_info' dictionary which is the standard MNE way
            raw_sub.info["subject_info"] = {"id": sub_id}

            raw_path = raw_dir / f"subj_{sub_id:03d}_raw.fif"
            events_path = event_dir / f"subj_{sub_id:03d}_events.npy"

            events_sub, _ = mne.events_from_annotations(raw_sub, event_id=event_id)
            np.save(events_path, events_sub)
            raw_sub.save(raw_path, overwrite=True)

            # Record metadata to CSV
            subject_rows.append(
                {
                    "subject_id": sub_id,
                    "raw_file": raw_path.name,
                    "events_file": events_path.name,
                }
            )

        pd.DataFrame(subject_rows).to_csv(subjects_csv, index=False)

    # 2. LOAD DATA FROM DISK
    with open(event_id_path, "r") as f:
        event_id = json.load(f)

    loaded_raws = []
    loaded_events = []
    subject_ids_out = []
    raw_filenames = []

    for sub_id in subjects:
        if sub_id in [88, 92, 100]:
            continue  # Skip subjects with known issues

        r_path = raw_dir / f"subj_{sub_id:03d}_raw.fif"
        e_path = event_dir / f"subj_{sub_id:03d}_events.npy"
        filename = f"subj_{sub_id:03d}_raw.fif"

        raw = mne.io.read_raw_fif(r_path, preload=True)
        evs = np.load(e_path)

        # Ensure the ID is present in the info after loading
        raw.info["subject_info"] = {"id": sub_id}

        loaded_raws.append(raw)
        loaded_events.append(evs)
        subject_ids_out.append(sub_id)
        raw_filenames.append(filename)

    return loaded_raws, loaded_events, event_id, subject_ids_out, raw_filenames


def load_target_subject_data(root: Path, source_path: Path, target_path: Path, resample: float | None) -> tuple:
    """
    Loads target data (Subject 110).
    1. Checks target_path for existing .fif files.
    2. If empty, processes XDF from source_path.
    3. Infers and saves event_id mapping.

    Parameters
    ----------
    root : Path
        The root directory of the BCI dataset.
    source_path : Path
        Path to the folder containing raw XDF files.
    target_path : Path
        Path to the folder where processed .fif files will be saved.

    Returns
    -------
    tuple
        A tuple containing:
        - loaded_raws: List of MNE Raw objects for each file.
        - loaded_events: List of event arrays for each file.
        - selected_event_id: Dictionary mapping event labels to event IDs.
        - subject_ids_out: List of subject IDs corresponding to the loaded data.
        - loaded_meta: Metadata dictionary containing filenames and channel names:
            {
                "filenames": raw_filenames,
                "channel_names": channel_names
            }

    Notes:
    -----
    The current implementation processes following types of recordings:
    1. Recordings with 16 channels and no channel information -> assign 16 channel labels from standard 1020 montage
    2. Recordings with 16 channels, and channel information exactly matches standard channel labels -> use existing channel labels
    3. Recordings with 17 channels -> P554 recordings with custom 16 channel labels + keyboard -> reorder channels to match standard
    4. Recordings with 24 channels -> discard the recording (different markers)

    P554 recordings of types "blinking", "jaw_clenching", "Music" are discarded.

    The channel labels are hardcoded in the _get_raw_xdf_offline function.
    A subset of channels can be selected during epoching.

    """

    raw_save_dir = target_path / "raws"
    event_save_dir = target_path / "events"
    event_id_path = target_path / "event_id.json"

    raw_save_dir.mkdir(parents=True, exist_ok=True)
    event_save_dir.mkdir(parents=True, exist_ok=True)

    # --- STEP 1: CHECK TARGET FOLDER ---
    existing_fif = sorted(list(raw_save_dir.glob("*.fif")))

    if len(existing_fif) > 0:
        print(f"Found {len(existing_fif)} processed files in target raw folder. Loading...")
        if resample is not None:
            print(f"Resampling to {resample} Hz...")
        loaded_raws = []
        loaded_events = []
        for fif_p in existing_fif:
            raw = mne.io.read_raw_fif(fif_p, preload=True)
            if resample is not None:
                sfreq = raw.info["sfreq"]
                if sfreq != resample:
                    original_sfreq = sfreq
                    raw.resample(resample)

            # Find corresponding event file
            npy_p = event_save_dir / f"{fif_p.stem.replace('_raw', '')}_events.npy"
            evs = np.load(npy_p)
            loaded_raws.append(raw)
            loaded_events.append(evs)

        with open(event_id_path, "r") as f:
            event_id = json.load(f)

        with open(target_path / "metadata.json", "r") as f:
            loaded_meta = json.load(f)

        return (
            loaded_raws,
            loaded_events,
            event_id,
            [110] * len(loaded_raws),
            loaded_meta,
        )

    # --- STEP 2: IF TARGET EMPTY, CHECK SOURCE ---
    if target_path is None:
        if source_path is None:
            raise ValueError(
                "Target folder is empty and no source_folder was provided to process raw XDF files."
            )
        else:
            print(
                f"Target folder is empty. Processing XDF files from source folder: {source_path}"
            )
        # process from source_path
    else:
        pass


    all_xdf = sorted([p for p in source_path.glob("*.xdf")])
    print(f"Found {len(all_xdf)} XDF files in source folder.")

    selected_files = all_xdf

    if not selected_files:
        raise FileNotFoundError(f"No XDF files found in {source_path}")

    loaded_raws, loaded_events, channel_names, raw_filenames = [], [], [], []
    selected_event_id = {}

    target_event_id = {
        "rest": 0,
        "left_hand": 1,
        "right_hand": 2,
    }

    # --- STEP 3: PROCESS XDF AND INFER EVENT_ID ---
    for file_path in selected_files:
        raw, markers, channel_labels = _get_raw_xdf_offline(file_path)

        if raw is None:
            print("raw is None, skipping the file.")
            continue
       

        montage = raw.info["chs"]
        
        selected_events = _standardize_and_map(raw, target_event_id)

        # NOTE: consider using JUMP and JUMP FAIL as evaluation of how user performs?

        selected_event_id.update(target_event_id)

        # Save files
        base_name = file_path.stem 
        raw.info["subject_info"] = {"id": 110}
        raw.save(raw_save_dir / f"{base_name}_raw.fif", overwrite=True)
        np.save(event_save_dir / f"{base_name}_events.npy", selected_events)

        loaded_raws.append(raw)  # list of MNE Raw objects
        loaded_events.append(selected_events)  # list of numpy arrays
        channel_names.append(channel_labels)  # list of channel name lists
        raw_filenames.append(f"{base_name}_raw")  # list of filenames

    # Save the inferred event_id for future runs
    with open(event_id_path, "w") as f:
        json.dump(selected_event_id, f, indent=2)

    loaded_meta = {
        "filenames": raw_filenames,
        "channel_names": channel_names
    }

    with open(target_path / "metadata.json", "w") as f:
        json.dump(loaded_meta, f, indent=2)

    return (
        loaded_raws,
        loaded_events,
        selected_event_id,
        [110] * len(loaded_raws),
        loaded_meta,
    )


def load_fina_baseline():
    pass


def load_physionet_baseline():
    pass


def create_subject_train_set(
    config: EEGConfig,
    all_raws: list,
    all_events: list,
    all_filenames: list,
    num_general: int = 0,
    num_dino: int = 0,
    num_supression: int = 0,
    shuffle: bool = True,
) -> tuple:
    """
    Create training dataset by selecting specific number of files from each category.

    Parameters
    ----------
    all_raws : list
        List of MNE Raw objects for all files
    all_events : list
        List of events arrays for all files
    all_filenames : list
        List of filenames (e.g., ["sub-P554_ses-S002_task-Default_run-001_eeg",
                                  "sub-P999_ses-S002_task-dino_run-001_eeg.xdf", ...])
    num_general : int
        Number of GENERAL (non-dino) files to use for training
    num_dino : int
        Number of DINO files to use for training
    num_supression: int
        Number of SUPRESSION files to use for training
    shuffle : bool
        Whether to randomly shuffle the files before selection

    Returns
    -------
    train_raws : list
        List of MNE Raw objects for training data
    train_events : ndarray
        List of training events
    train_filenames : list
        List of filenames used in training
    train_sub_ids : list
        List of subject IDs corresponding to training data
    used_indices : list
        Indices of files used (for excluding from test set)

    Notes
    -----
    The function randomly selects the specified number of files from each category
    to form the training dataset if shuffle is True.

    Total number of GENERAL files available: 
    Total number of DINO files available: 
    Total number of SUPRESSION files available: 

    """
    np.random.seed(config.random_state)

    # Categorize files by type
    general_indices = [i for i, fname in enumerate(all_filenames) if ("dino" not in fname.lower() and "s001" not in fname.lower()
                                                                      and "supression" not in fname.lower())]
    print(f"Total number of GENERAL files available: {len(general_indices)}")
    dino_indices = [
        i
        for i, fname in enumerate(all_filenames)
        if ("dino" in fname.lower() or "s001" in fname.lower()) and "supression" not in fname.lower()
    ]
    print(f"Total number of DINO files available: {len(dino_indices)}")

    supression_indices = [i for i, fname in enumerate(all_filenames) if "supression" in fname.lower()]     # one file only
    print(f"Total number of SUPRESSION files available: {len(supression_indices)}")

    # Randomly shuffle
    if shuffle:
        np.random.shuffle(general_indices)
        np.random.shuffle(dino_indices)

    if len(general_indices) < num_general:
        raise ValueError(
            f"Not enough general files available for training. "
            f"Available: {len(general_indices)}, Requested: {num_general}"
        )
    if len(dino_indices) < num_dino:
        raise ValueError(
            f"Not enough dino files available for training. "
            f"Available: {len(dino_indices)}, Requested: {num_dino}"
        )
    if len(supression_indices) < num_supression:
        raise ValueError(
            f"Not enough supression files available for training. "
            f"Available: {len(supression_indices)}, Requested: {num_supression}"
        )

    # Select training files
    train_indices = (
        general_indices[:num_general]
        + dino_indices[:num_dino]
        + supression_indices[:num_supression]
    )

    if not train_indices:
        raise ValueError("No files selected for training. Check num_ values.")

    print(f"\n=== Training Dataset, Target Subject ===")
    print(f"Selected {len(train_indices)} files:")
    print(f"  - General: {len(general_indices[:num_general])}")
    print(
        f"  - Dino: {len(dino_indices[:num_dino])}"
    )
    print(f"  - Supression: {len(supression_indices[:num_supression])}")

    # Build training set
    train_raws = [all_raws[i] for i in train_indices]
    train_events = [all_events[i] for i in train_indices]
    train_filenames = [all_filenames[i] for i in train_indices]
    train_sub_ids = [110] * len(train_raws)  # All from subject 110

    print(f"Training files: {train_filenames}")

    return (
        train_raws,
        train_events,
        train_filenames,
        train_sub_ids,
        train_indices,  # Return used indices to exclude from test
    )


def create_subject_test_set(
    config: EEGConfig,
    all_raws: list,
    all_events: list,
    all_filenames: list,
    exclude_indices: list,
    num_general: int,
    num_dino: int,
    num_supression: int,
    shuffle: bool = False,
) -> tuple:
    """
    Create testing dataset from files NOT used in training.
    Selects exactly the specified number of files from each category.

    Parameters
    ----------
    all_raws : list
        List of MNE Raw objects for all files
    all_events : list
        List of events arrays for all files
    all_filenames : list
        List of filenames
    exclude_indices : list
        Indices of files already used in training (to exclude). Can be empty.
    num_general : int
        Exact number of GENERAL files for testing
    num_dino : int
        Exact number of DINO files for testing
    num_supression: int
        Exact number of SUPRESSION files for testing
    shuffle : bool
        Whether to randomly shuffle the files before selection

    Returns
    -------
    test_raws : list
        List of MNE Raw objects for testing data
    test_events : ndarray
        Test events
    test_filenames : list
        List of filenames used in testing
    test_sub_ids : list
        List of subject IDs corresponding to testing data

    Notes
    -----
        Total number of GENERAL files available: 
        Total number of DINO files available: 
        Total number of SUPRESSION files available: 
    """
    np.random.seed(config.random_state)

    # Get available (unused) indices for each category
    available_indices = [
        i for i in range(len(all_filenames)) if i not in exclude_indices
    ]

    general_available = [i for i in available_indices if ("dino" not in all_filenames[i].lower()
                         and "s001" not in all_filenames[i].lower() and "supression" not in all_filenames[i].lower())]
    dino_available = [
        i
        for i in available_indices
        if (("dino" in all_filenames[i].lower() or "s001" in all_filenames[i].lower()) and "supression" not in all_filenames[i].lower())
    ]
    supression_available = [i for i in available_indices if "supression" in all_filenames[i].lower()]   # one file only

    # Check if enough files are available
    if len(general_available) < num_general:
        raise ValueError(
            f"Not enough general files available for testing. "
            f"Available: {len(general_available)}, Requested: {num_general}"
        )
    if len(dino_available) < num_dino:
        raise ValueError(
            f"Not enough dino files available for testing. "
            f"Available: {len(dino_available)}, Requested: {num_dino}"
        )
    if len(supression_available) < num_supression:
        raise ValueError(
            f"Not enough supression files available for testing. "
            f"Available: {len(supression_available)}, Requested: {num_supression}" 
        )

    # Shuffle and select exact number
    if shuffle:
        np.random.shuffle(general_available)
        np.random.shuffle(dino_available)

    test_indices = (
        general_available[:num_general]
        + dino_available[:num_dino]
        + supression_available[:num_supression]
    )

    if not test_indices:
        raise ValueError("No files selected for testing.")

    print(f"\n=== Testing Dataset, Target Subject ===")
    print(f"Selected {len(test_indices)} files:")
    print(f"  - General: {num_general}")
    print(f"  - Dino: {num_dino}")
    print(f"  - Supression: {num_supression}")

    # Build test set
    test_raws = [all_raws[i] for i in test_indices]
    test_events = [all_events[i] for i in test_indices]
    test_filenames = [all_filenames[i] for i in test_indices]
    test_sub_ids = [110] * len(test_raws)  # All from subject 110

    print(f"Test files: {test_filenames}")
    
    return (
        test_raws,
        test_events,
        test_filenames,
        test_sub_ids
    )

# TODO: delete in final version 
# def load_cho2017_data(subjects: list[int], root: Path, channels: list[str]) -> tuple:
#     """Load Physionet Motor Imagery data for specified subjects.

#     This function checks if the data is already saved on disk. If not, it downloads
#     the data from Physionet, preprocesses it (attaching subject IDs), and saves it
#     to disk for future use.

#     Parameters
#     ----------
#     subjects : list[int]
#         List of subject IDs to load. IDs are integers.
#     root : str
#         Path to the root directory of the dataset.
#     channels : list[str]
#         List of channel names to load from the raw data.

#     Returns
#     -------
#     tuple
#         A tuple containing:
#         - loaded_raws: List of MNE Raw objects for each subject.
#         - loaded_events: List of event arrays for each subject.
#         - event_id: Dictionary mapping event labels to event IDs.
#         - subject_ids_out: List of subject IDs corresponding to the loaded data.
#         - raw_filenames: List of raw filenames corresponding to the loaded data.
#     """

#     cho2017_root = root / "data" / "datasets" / "cho2017"
#     raw_dir = cho2017_root / "raws"
#     event_dir = cho2017_root / "events"
#     meta_dir = cho2017_root / "metadata"
#     event_id_path = cho2017_root / "event_id.json"
#     subjects_csv = meta_dir / "subjects.csv"

#     for d in [raw_dir, event_dir, meta_dir]:
#         d.mkdir(parents=True, exist_ok=True)

#     # 1. DATA GENERATION / SAVING TO DISK
#     if not subjects_csv.exists():
#         dataset = Cho2017()
#         # print("Dataset subjects available:", dataset.subject_list)
#         if subjects:
#             target_subjects = subjects
#         else:
#             target_subjects = dataset.subject_list
#         # dataset.feet_runs = []

#         # remove = [88, 92, 100]
#         # target_subjects = [x for x in dataset.subject_list if x not in remove]

#         # raise KeyboardInterrupt("Need to specify target subjects for Cho2017 dataset.")
#         # event_dict = dataset.events # (other integers: 1, 2, 3)
#         # print(event_dict)
#         event_id = {
#             "left_hand": 1,
#             "right_hand": 2,
#         }           # Cho2017 data does not have rest events

#         # Save event_id mapping
#         with open(event_id_path, "w") as f:
#             json.dump(event_id, f, indent=2)

#         print("Processing and saving Cho2017 data to disk...")
#         # print("Keeping all 64 channels for now...")
        
#         subject_rows = []

#         for sub_id in target_subjects:
#             raw_dict = dataset.get_data(subjects=[sub_id])
#             raw_list = []
#             for session in raw_dict[sub_id]:
#                 for run in raw_dict[sub_id][session]:
#                     # print(raw_dict[sub_id][session][run])
#                     raw = raw_dict[sub_id][session][run]
#                     raw_list.append(raw_dict[sub_id][session][run])

#             raw_sub = mne.concatenate_raws(raw_list)
            
#             if channels is not None:
#                 raw_sub.pick(channels)
#                 raw_sub.reorder_channels(channels)
#             else:
#                 print("Keeping all 64 channels for now...")

#             raw_sub.resample(160)

#             # --- ATTACH SUBJECT ID TO RAW ---
#             # Use the 'subject_info' dictionary which is the standard MNE way
#             raw_sub.info["subject_info"] = {"id": sub_id}

#             raw_path = raw_dir / f"subj_{sub_id:03d}_raw.fif"
#             events_path = event_dir / f"subj_{sub_id:03d}_events.npy"

#             # events_sub, _ = mne.events_from_annotations(raw_sub)
#             # print(original_event_id)

#             events_sub, _ = mne.events_from_annotations(raw_sub, event_id=event_id)
#             np.save(events_path, events_sub)
#             raw_sub.save(raw_path, overwrite=True)
#             # np.save(events_path, mne.find_events(raw_sub))

#             # Record metadata to CSV
#             subject_rows.append(
#                 {
#                     "subject_id": sub_id,
#                     "raw_file": raw_path.name,
#                     "events_file": events_path.name,
#                 }
#             )

#         pd.DataFrame(subject_rows).to_csv(subjects_csv, index=False)
    
#     #raise KeyboardInterrupt("Need to specify target subjects for Cho2017 dataset.")

#     # 2. LOAD DATA FROM DISK
#     with open(event_id_path, "r") as f:
#         event_id = json.load(f)

#     loaded_raws = []
#     loaded_events = []
#     subject_ids_out = []
#     raw_filenames = []

#     for sub_id in subjects:

#         r_path = raw_dir / f"subj_{sub_id:03d}_raw.fif"
#         e_path = event_dir / f"subj_{sub_id:03d}_events.npy"
#         filename = f"subj_{sub_id:03d}_raw.fif"

#         raw = mne.io.read_raw_fif(r_path, preload=True)
#         evs = np.load(e_path)

#         # Ensure the ID is present in the info after loading
#         raw.info["subject_info"] = {"id": sub_id}

#         loaded_raws.append(raw)
#         loaded_events.append(evs)
#         subject_ids_out.append(sub_id)
#         raw_filenames.append(filename)

#     return loaded_raws, loaded_events, event_id, subject_ids_out, raw_filenames


# # loaded_cho_raws, loaded_cho_events, cho_event_id, cho_subject_ids, cho_raw_filenames = load_cho2017_data(
# #     subjects = [],  # Cho2017 has subjects 1 to 54
# #     root = Path.cwd(),
# #     channels = [ "Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz", "C4", "T8", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]
# # )



# def create_cho_epochs(filtered_raw, events, sub_id, filename, original_event_id, target_event_id):

#     # complete_epochs = []
#     # Create the epochs for CV with metadata
#     mi_epochs = mne.Epochs(
#         filtered_raw,
#         events,
#         event_id=original_event_id,
#         tmin=0.5,
#         tmax=4.0,
#         preload=True,
#         baseline=None,
#     )

#     # Attach metadata
#     mi_metadata = pd.DataFrame(
#         {
#             "subject_id": [sub_id] * len(mi_epochs),
#             "filename": [filename] * len(mi_epochs),
#             "condition": mi_epochs.events[:, 2],
#         }
#     )

#     mi_epochs.metadata = mi_metadata
#     # 2. Create REST epochs from pre-trial period
#     # We'll create synthetic "rest" events BEFORE each MI event
#     rest_events = events.copy()
    
#     # Shift events backward by 4 seconds (so rest ends where MI begins)
#     # This ensures rest epochs are [-3.5, 0.5]s relative to MI onset
#     # But we'll extract [0.5, 4.0]s from these shifted events
#     rest_events[:, 0] = rest_events[:, 0] - int(4.0 * filtered_raw.info['sfreq'])
#     rest_events[:, 2] = 0  # Label all as "rest"
    
#     # Filter out events that would go before recording start
#     valid_rest_events = rest_events[rest_events[:, 0] >= int(0.5 * filtered_raw.info['sfreq'])]
    
#     rest_epochs = mne.Epochs(
#         filtered_raw,
#         valid_rest_events,
#         event_id={'rest': 0},
#         tmin=0.5,
#         tmax=4.0,  # Same time window as MI epochs!
#         preload=True,
#         baseline=None,
#     )
    
#     print(f"Rest epochs shape: {rest_epochs.get_data().shape}")


#     # TODO: Normalize the data
     
#     rest_epochs.event_id = {'rest': 0}
#     # Attach metadata
#     rest_metadata = pd.DataFrame(
#         {
#             "subject_id": [sub_id] * len(rest_epochs),
#             "filename": [filename] * len(rest_epochs),
#             "condition": rest_epochs.events[:, 2],
#         }
#     )
#     rest_epochs.metadata = rest_metadata
#     print(rest_epochs.copy().get_data().shape)

#     # Keep MI labels as 1 and 2
#     mi_epochs.event_id = original_event_id


#     # Combine
#     complete_epochs = mne.concatenate_epochs([rest_epochs, mi_epochs])

#     # Final event_id
#     complete_epochs.event_id = target_event_id

#     return complete_epochs



def validate_train_test_split(train_filenames: list, test_filenames: list) -> None:
    """
    Validates that training and testing datasets have no overlapping files.
    
    Parameters
    ----------
    train_filenames : list
        List of filenames used for training
    test_filenames : list
        List of filenames used for testing
    
    Raises
    ------
    ValueError
        If any files appear in both training and testing sets
    
    Notes
    -----
    Also prints summary statistics about the split.
    """
    print("\n" + "=" * 80)
    print("VALIDATING TRAIN/TEST SPLIT")
    print("=" * 80)
    
    # Convert to sets for efficient intersection
    train_set = set(train_filenames)
    test_set = set(test_filenames)
    
    # Check for overlaps
    overlap = train_set & test_set
    
    if overlap:
        print("❌ ERROR: Found overlapping files between train and test sets!")
        print(f"\nNumber of overlapping files: {len(overlap)}")
        print("\nOverlapping files:")
        for fname in sorted(overlap):
            print(f"  - {fname}")
        
        raise ValueError(
            f"Data leakage detected! {len(overlap)} file(s) appear in both "
            f"training and testing sets. This will invalidate your results."
        )
    
    # Print summary
    print("✅ No overlap detected between train and test sets")
    print(f"\nDataset Statistics:")
    print(f"  Training files:   {len(train_filenames)}")
    print(f"  Testing files:    {len(test_filenames)}")
    print(f"  Total unique:     {len(train_set | test_set)}")
    print(f"  Expected total:   {len(train_filenames) + len(test_filenames)}")
    
    # Categorize by type
    train_categories = {
        "general": [f for f in train_filenames 
                   if "dino" not in f.lower() and "s001" not in f.lower() and "supression" not in f.lower()],
        "dino": [f for f in train_filenames 
                if ("dino" in f.lower() or "s001" in f.lower()) and "supression" not in f.lower()],
        "supression": [f for f in train_filenames if "supression" in f.lower()],
    }
    
    test_categories = {
        "general": [f for f in test_filenames 
                   if "dino" not in f.lower() and "s001" not in f.lower() and "supression" not in f.lower()],
        "dino": [f for f in test_filenames 
                if ("dino" in f.lower() or "s001" in f.lower()) and "supression" not in f.lower()],
        "supression": [f for f in test_filenames if "supression" in f.lower()],
    }
    
    print(f"\nTraining Set Composition:")
    print(f"  General:     {len(train_categories['general'])}")
    print(f"  Dino:        {len(train_categories['dino'])}")
    print(f"  Supression:  {len(train_categories['supression'])}")
    
    print(f"\nTesting Set Composition:")
    print(f"  General:     {len(test_categories['general'])}")
    print(f"  Dino:        {len(test_categories['dino'])}")
    print(f"  Supression:  {len(test_categories['supression'])}")
    
    # Warn about small test sets
    if len(test_filenames) < 2:
        print("\n⚠️ WARNING: Test set has fewer than 2 files. Results may not be reliable.")
    
    if len(train_filenames) < 3:
        print("\n⚠️ WARNING: Train set has fewer than 3 files. Model may not generalize well.")
    
    print("=" * 80 + "\n")
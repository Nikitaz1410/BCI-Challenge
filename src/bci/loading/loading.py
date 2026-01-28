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
    The current implementation processes following types of recordings:
    1. Recordings with 16 channels and no channel information -> assign 16 channel labels from standard 1020 montage
    2. Recordings with 16 channels, and channel information exactly matches standard channel labels -> use existing channel labels
    3. Recordings with 17 channels -> P554 recordings with custom 16 channel labels + keyboard -> reorder channels to match standard
    4. Recordings with 24 channels -> discard the recording (different markers)

    P554 recordings of types "blinking", "jaw_clenching", "Music" are discarded.
    """
    import pyxdf

    print(("=" * 30) + f" Processing file: {trial.name} " + ("=" * 30))

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

    # Iterate through streams to find desired indices, extract EEG stream
    for i, stream in enumerate(streams):
        name = stream["info"]["name"][0]  # name of the stream
        if name == "EEG - Impedances":
            continue
            _index_impedance = i  # impedance stream
        elif name == "EEG":
            eeg_channel = i  # eeg stream
        elif name == "pupil_capture_pupillometry_only":
            continue
            pupil_channel = i  # pupil stream
        elif name == "pupil_capture":
            continue
            pupil_capture_channel = i  # pupil stream
        elif name == "pupil_capture_fixations":
            continue
            pupil_capture_fixations_channel = i
        else:
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
            # print("Discarding this recording for further processing...")
            # return None, None, None
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
        if "P554" in trial.name:
            print(
                "P554 recording detected. Assigning custom 16 channel labels + keyboard..."
            )
            custom_channels = [
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
                custom_channels  # Custom channel labels for P554 with 16 channels
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

    # print(len(data), data.shape)

    # Verify we have the correct number of channels
    if data.shape[0] != len(channel_labels):
        print(
            f"Data has {data.shape[0]} channels, but expected {len(channel_labels)} channels."
        )
        print("Discarding this recording from further processing...")
        return None, None, None

    # print(f"Final channel labels used ({len(channel_labels)} channels):", channel_labels)

    # Get sampling frequency and create info object
    sfreq = float(streams[eeg_channel]["info"]["nominal_srate"][0])
    info = mne.create_info(channel_labels, sfreq, ch_types="eeg")

    print("channelds from info:", info["ch_names"])

    # Create raw object and set the montage
    raw_data = mne.io.RawArray(data, info, verbose=False)
    raw_data.set_montage(montage)

    if "P554" in trial.name:
        print("Reordering channels for standard 10-20 montage...")
        raw_data.reorder_channels(standard_channels)
        channel_labels = raw_data.ch_names

    print(
        f"Final channel labels used ({len(raw_data.ch_names)} channels):",
        raw_data.ch_names,
    )

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


def _standardize_and_map(raw, target_event_id, mode="general"):
    """
    Standardizes markers based on the task type (dino or general).

    Args:
        raw: MNE Raw object.
        target_event_id: The final integer mapping,
        {
            "rest": 0,
            "left_hand": 1,
            "right_hand": 2
        }
        mode: "general" or "dino"

    NOTE: For Dino, we have different events, so we need to handle them separately
        ''' Events IDs for Dino files
        {np.str_(''): 1, np.str_('ARROW LEFT ONSET'): 2,
         np.str_('ARROW RIGHT ONSET'): 3,
         np.str_('CIRCLE ONSET'): 4,
         np.str_('JUMP'): 5,
         np.str_('JUMP FAIL'): 6}
         '''
    """

    # 1. Define mappings specific to each recording type
    if mode == "dino":
        rename_map = {
            "ARROW LEFT ONSET": "left_hand",
            "ARROW RIGHT ONSET": "right_hand",
            "CIRCLE ONSET": "rest",
        }  # For now, ignore JUMP and FAIL markers
    else:  # "general" mode
        rename_map = {
            "ARROW LEFT": "left_hand",
            "ARROW RIGHT": "right_hand",
            "CIRCLE": "rest",
        }

    # 2. Apply the rename to the raw object
    existing_descriptions = set(raw.annotations.description)
    # print("Existing Annotations before renaming:", existing_descriptions)

    actual_map = {k: v for k, v in rename_map.items() if k in existing_descriptions}
    raw.annotations.rename(actual_map)

    # 3. Force extraction of ONLY the standardized keys
    events, _ = mne.events_from_annotations(raw, event_id=target_event_id)
    if len(events) == 0:
        raise ValueError(f"No valid events found after standardization in {mode} mode")

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
        # print("Keeping all 64 channels for now...")

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
            # np.save(events_path, mne.find_events(raw_sub))

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


def load_target_subject_data(root: Path, source_path: Path, target_path: Path) -> tuple:
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
        print(f"Found {len(existing_fif)} processed files in target folder. Loading...")

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

        with open(event_id_path, "r") as f:
            event_id = json.load(f)
            # print(event_id)

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
        raise ValueError(
            "Target folder is empty and no source_folder was provided to process raw XDF files."
        )
    else:
        print(
            f"Target folder is empty. Processing XDF files from source folder: {source_path}"
        )

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
        # TODO: new check
        sfreq = raw.info["sfreq"]
        print(f"  Original sampling rate: {sfreq} Hz")

        if abs(sfreq - 160) > 1:
            print(f"  Resampling to 160 Hz...")
            raw.resample(160)
        else:
            print(f"  Sampling rate already 160 Hz, skipping resample")

        # raw.resample(160)

        task_mode = "dino" if "dino" in file_path.name.lower() else "general"
        # Standardize and map using the specific mode
        selected_events = _standardize_and_map(raw, target_event_id, mode=task_mode)

        # NOTE: consider using JUMP and JUMP FAIL as evaluation of how user performs?

        selected_event_id.update(target_event_id)

        # Save files
        base_name = file_path.stem  # filename without extension
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


# NOTE: The following is an example of how to use the loaded data for you guys as a reference
# on how the data could be processed further.
# TODO: Needs to be removed from the final version.


def create_subject_train_set(
    config: EEGConfig,
    all_raws: list,
    all_events: list,
    all_filenames: list,
    num_p554: int = 0,
    num_p999_general: int = 0,
    num_p999_dino: int = 0,
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
    num_p554 : int
        Number of P554 files to use for training
    num_p999_general : int
        Number of "P999-general" (non-dino) files to use for training
    num_p999_dino : int
        Number of P999-dino files to use for training
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

    Total number of P554 files available: 2
    Total number of P999-general files available: 4
    Total number of P999-dino files available: 13

    """
    np.random.seed(config.random_state)

    # Categorize files by type
    p554_indices = [i for i, fname in enumerate(all_filenames) if "P554" in fname]
    print(f"Total number of P554 files available: {len(p554_indices)}")
    p999_general_indices = [
        i
        for i, fname in enumerate(all_filenames)
        if "P999" in fname and "dino" not in fname.lower()
    ]
    print(f"Total number of P999-general files available: {len(p999_general_indices)}")
    p999_dino_indices = [
        i
        for i, fname in enumerate(all_filenames)
        if "P999" in fname and "dino" in fname.lower()
    ]
    print(f"Total number of P999-dino files available: {len(p999_dino_indices)}")

    # Randomly shuffle
    if shuffle:
        np.random.shuffle(p554_indices)
        np.random.shuffle(p999_general_indices)
        np.random.shuffle(p999_dino_indices)

    if len(p554_indices) < num_p554:
        raise ValueError(
            f"Not enough P554 files available for training. "
            f"Available: {len(p554_indices)}, Requested: {num_p554}"
        )
    if len(p999_general_indices) < num_p999_general:
        raise ValueError(
            f"Not enough P999-general files available for training. "
            f"Available: {len(p999_general_indices)}, Requested: {num_p999_general}"
        )
    if len(p999_dino_indices) < num_p999_dino:
        raise ValueError(
            f"Not enough P999-dino files available for training. "
            f"Available: {len(p999_dino_indices)}, Requested: {num_p999_dino}"
        )

    # Select training files
    train_indices = (
        p554_indices[:num_p554]
        + p999_general_indices[:num_p999_general]
        + p999_dino_indices[:num_p999_dino]
    )

    if not train_indices:
        raise ValueError("No files selected for training. Check num_ values.")

    print(f"\n=== Training Dataset, Target Subject ===")
    print(f"Selected {len(train_indices)} files:")
    print(f"  - P554: {len(p554_indices[:num_p554])}")
    print(
        f"  - P999-general (non-dino): {len(p999_general_indices[:num_p999_general])}"
    )
    print(f"  - P999-dino: {len(p999_dino_indices[:num_p999_dino])}")

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
    num_p554: int,
    num_p999_general: int,
    num_p999_dino: int,
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
    num_p554 : int
        Exact number of P554 files to use for testing
    num_p999_general : int
        Exact number of P999-general files for testing
    num_p999_dino : int
        Exact number of P999-dino files for testing
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
        Total number of P554 files available: 2
        Total number of P999-general files available: 4
        Total number of P999-dino files available: 13
    """
    np.random.seed(config.random_state)

    # Get available (unused) indices for each category
    available_indices = [
        i for i in range(len(all_filenames)) if i not in exclude_indices
    ]

    p554_available = [i for i in available_indices if "P554" in all_filenames[i]]
    p999_general_available = [
        i
        for i in available_indices
        if "P999" in all_filenames[i] and "dino" not in all_filenames[i].lower()
    ]
    p999_dino_available = [
        i
        for i in available_indices
        if "P999" in all_filenames[i] and "dino" in all_filenames[i].lower()
    ]

    # Check if enough files are available
    if len(p554_available) < num_p554:
        raise ValueError(
            f"Not enough P554 files available for testing. "
            f"Available: {len(p554_available)}, Requested: {num_p554}"
        )
    if len(p999_general_available) < num_p999_general:
        raise ValueError(
            f"Not enough P999-general files available for testing. "
            f"Available: {len(p999_general_available)}, Requested: {num_p999_general}"
        )
    if len(p999_dino_available) < num_p999_dino:
        raise ValueError(
            f"Not enough P999-dino files available for testing. "
            f"Available: {len(p999_dino_available)}, Requested: {num_p999_dino}"
        )

    # Shuffle and select exact number
    if shuffle:
        np.random.shuffle(p554_available)
        np.random.shuffle(p999_general_available)
        np.random.shuffle(p999_dino_available)

    test_indices = (
        p554_available[:num_p554]
        + p999_general_available[:num_p999_general]
        + p999_dino_available[:num_p999_dino]
    )

    if not test_indices:
        raise ValueError("No files selected for testing.")

    print(f"\n=== Testing Dataset, Target Subject ===")
    print(f"Selected {len(test_indices)} files:")
    print(f"  - P554: {num_p554}")
    print(f"  - P999-general (non-dino): {num_p999_general}")
    print(f"  - P999-dino: {num_p999_dino}")

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
import numpy as np
import mne
from pathlib import Path
import pyxdf


def get_raw_xdf_offline(
    trial: Path, marker_durations: list[float] | None = None
) -> tuple[mne.io.RawArray, list, list[str]]:
    """
    Function to load the raw data from the trial and return the raw data object, the markers and the channel labels.
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
    
    Number of channels should be 16, the order and naming of channels should match the list of channel_labels below.
    Function may need to be adapted for other recordings.
    """

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








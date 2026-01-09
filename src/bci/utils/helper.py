# Authors: BCI chair

import numpy as np
import mne
from pathlib import Path
import pyxdf


def get_raw_offline(
    trial: Path, marker_durations: list[float] | None = None
) -> tuple[mne.io.RawArray, list, list[str]]:
    """
    Function to load the raw data from the trial and return the raw data object, the markers and the channel labels.

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
    """
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

    # Get the channel labels
    # channels = streams[eeg_channel]["info"]["desc"][0]["channels"][0]["channel"]
    # channel_labels = [channel["label"][0] for channel in channels]
    channel_labels = [
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
        "keyboard",
    ]  # Manually set for g.Nautilus

    # same for pupil channel
    if pupil_channel is not None:
        pupil_channels = streams[pupil_channel]["info"]["desc"][0]["channels"][0][
            "channel"
        ]
        pupil_labels = [channel["label"][0] for channel in pupil_channels]
        print("Pupil channels:", pupil_labels)
    if pupil_capture_channel is not None:
        pupil_capture_channels = streams[pupil_capture_channel]["info"]["desc"][0][
            "channels"
        ][0]["channel"]
        pupil_capture_labels = [
            channel["label"][0] for channel in pupil_capture_channels
        ]
        print("Pupil capture channels:", pupil_capture_labels)
    if pupil_capture_fixations_channel is not None:
        pupil_capture_fixations_channels = streams[pupil_capture_fixations_channel][
            "info"
        ]["desc"][0]["channels"][0]["channel"]
        pupil_capture_fixations_labels = [
            channel["label"][0] for channel in pupil_capture_fixations_channels
        ]
        print("Pupil capture fixations channels:", pupil_capture_fixations_labels)

    # Create montage object - this is needed for the raw data object (Layout of the electrodes)
    """
    montage_dict = {
        channel["label"][0]: [float(channel["location"][0][dim][0]) for dim in "XYZ"]
        for channel in channels
    }
    montage = mne.channels.make_dig_montage(ch_pos=montage_dict, coord_frame="head")
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    ts = streams[eeg_channel]["time_series"][-1]
    print(
        f"EEG data timestamps range from {ts[0]/250.0} to {ts[-1]/250.0} (total {ts[-1]/250.0-ts[0]/250.0:.2f} seconds)"
    )

    # Get EEG data - https://mne.tools/dev/auto_examples/io/read_xdf.html#ex-read-xdf
    data = streams[eeg_channel]["time_series"].T * 1e-6  # scaling the data to volts

    # exclude last channel if it is impedance
    if channel_labels[-1].lower() in ["ts", "impedances", "keyboard"]:
        data = data[:-1, :]
        channel_labels = channel_labels[:-1]
    # Get sampling frequency and create info object

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


def get_epochs(
    raw_data: mne.io.Raw,
    markers: list,
    tmin: float = 0.3,
    tmax: float = 1.3,
    baseline: tuple[float, float] | None = None,
    event_dict: dict | None = None,
) -> tuple[mne.Epochs, dict]:
    """
    Get the epochs from the raw data based on the markers.

    Parameters
    ----------
    raw_data : mne.io.RawArray
        Raw data object.
    markers : list
        List of markers.
    tmin : float, optional
        Start time of the time window (epoch). The default is 0.3.
    tmax : float, optional
        End time of the time window (epoch). The default is 1.3.
    baseline : tuple, optional
        Window used for baseline correction. The default is None.
    event_dict : dict, optional
        Dictionary of event IDs. The default is None.
    Returns
    -------
    epochs : mne.Epochs
        Epochs object.
    filtered_events_id : dict
        Dictionary of filtered events.

    """
    # Get the events based on the annotations (in our case the ARROW markers)
    markers_dict = {marker: i for i, marker in enumerate(markers)}
    if event_dict is not None:
        events, events_id = mne.events_from_annotations(
            raw_data, event_id=event_dict, verbose=False
        )
    else:
        events, events_id = mne.events_from_annotations(
            raw_data, event_id=markers_dict, verbose=False
        )

    # Get the events that are in the marker_IDs
    filtered_events_id = {key: value for key, value in events_id.items() if "" in key}
    tmin_ = None

    # Adapt time window in case we need to do baseline correction
    if baseline:
        tmin_ = tmin
        tmin = baseline[0]
    else:
        baseline = None

    # Create epochs
    epochs = mne.Epochs(
        raw_data,
        events=events,
        tmin=tmin,
        tmax=tmax,
        event_id=filtered_events_id,
        baseline=baseline,
        preload=True,
        verbose=False,
        event_repeated="drop",
    )

    # Crop the epochs to the desired time window
    if tmin_ is not None:
        epochs.crop(tmin=tmin_, tmax=tmax)

    return epochs, events, filtered_events_id


from mne.time_frequency import psd_array_multitaper


def compute_psd(data, fmin, fmax):
    psd, freqs = psd_array_multitaper(
        data, sfreq=250, fmin=fmin, fmax=fmax, verbose=False
    )
    return psd


from mne.decoding import CSP

# class for computing selected features training and inference
from sklearn.base import BaseEstimator, TransformerMixin


class compute_features(BaseEstimator, TransformerMixin):
    def __init__(self, fmin=8, fmax=30):
        self.fmin = fmin
        self.fmax = fmax
        self.csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    def fit(self, X, y=None):
        self.csp.fit(X, y)
        return self

    def transform(self, X):
        psd_features = compute_psd(X, fmin=self.fmin, fmax=self.fmax)
        psd_features = psd_features.reshape(
            psd_features.shape[0], -1
        )  # Flatten the PSD features
        csp_features = self.csp.transform(X)

        features = np.concatenate((psd_features, csp_features), axis=-1)
        return features

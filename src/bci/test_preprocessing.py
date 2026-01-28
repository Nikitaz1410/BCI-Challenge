import sys
import scipy

import matplotlib.pyplot as plt

from pathlib import Path

import mne
import numpy as np
import pyxdf

from bci.preprocessing.filters import (
    Filter,
)
from bci.utils.bci_config import load_config

# FINA's Data
current_wd = Path.cwd()  # BCI-Challenge directory
trial = (
    current_wd / "data" / "sub-P554" / "sub-P554_ses-S002_task-Default_run-001_eeg.xdf"
)
# trial = (current_wd / "data" / "sub-P999" / "sub-P999_ses-S002_task-arrow_run-002_eeg.xdf")

streams, header = pyxdf.load_xdf(trial, verbose=False)

event_channel = None
markers = []

pupil_capture_channel, pupil_capture_fixations_channel, pupil_channel = (
    None,
    None,
    None,
)
eeg_channel = None

marker_durations = [3, 1, 3]

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
]

# Create montage object - this is needed for the raw data object (Layout of the electrodes)
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

print(len(data), data.shape)

# Verify we have the correct number of channels
if data.shape[0] != len(channel_labels):
    print(
        f"Data has {data.shape[0]} channels, but expected {len(channel_labels)} channels"
    )

# Get sampling frequency and create info object
sfreq = float(streams[eeg_channel]["info"]["nominal_srate"][0])
info = mne.create_info(channel_labels, sfreq, ch_types="eeg")

# Create raw object and set the montage
raw_data = mne.io.RawArray(data, info, verbose=False)
raw_data.set_montage(montage)

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


# # Plot the raw signal with annotations
# raw_data.plot(
#     n_channels=len(channel_labels),
#     scalings="auto",
#     title="Raw EEG Signal with Annotations",
#     block=True,
# )

raw_data.resample(160)

# # Plot the PSD of the lfilter filtered raw
raw_data.compute_psd(method="welch", fmax=70).plot(
    picks="eeg", average=False, spatial_colors=True, dB=True
)

# This command opens the window to display the plot
plt.show()

try:
    config_path = current_wd / "resources" / "configs" / "bci_config.yaml"
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    print("Configuration loaded successfully!")
except Exception as e:
    print(f"❌ Error loading config: {e}")
    sys.exit(1)

# I need to filter the data
filt = Filter(config, online=False)


# This command opens the window to display the plot
plt.show()

filtered_raw = filt.apply_filter_offline(raw_data)
# # Design a Butterworth filter using lfilter
# b, a = scipy.signal.butter(
#     config.order,
#     Wn=np.array(config.frequencies),
#     btype="bandpass",
#     fs=config.fs,
# )

# filtered_signal_lfilter = scipy.signal.lfilter(b, a, raw_data.get_data())

# # Create a new Raw object with the lfilter filtered data
# filtered_raw_lfilter = raw_data.copy()
# filtered_raw_lfilter._data = filtered_signal_lfilter

# # Plot the PSD of the lfilter filtered raw
# filtered_raw_lfilter.compute_psd(method="welch", fmax=70).plot(
#     picks="eeg",
#     average=False,
#     spatial_colors=True,
#     dB=True,
# )

# This command opens the window to display the plot
plt.show()

# # Plot the raw signal with annotations for lfilter
# filtered_raw_lfilter.plot(
#     n_channels=len(channel_labels),
#     scalings="auto",
#     title="Filtered Raw EEG Signal with Annotations (lfilter)",
#     block=True,
# )

# Plot the PSD of the filtered_raw
filtered_raw.compute_psd(method="welch", fmax=70).plot(
    picks="eeg",
    average=False,
    spatial_colors=True,
    dB=True,
)

# This command opens the window to display the plot
plt.show()

# # Plot the raw signal with annotations
# filtered_raw.plot(
#     n_channels=len(channel_labels),
#     scalings="auto",
#     title="Filtered Raw EEG Signal with Annotations",
#     block=True,
# )

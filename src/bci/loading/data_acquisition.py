import mne
import scipy

from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# Extract the data from eegbci
# Need baseline data (Rest with eyes open) and Motor Imagery Data (Left/Right Hand)


def open_eeg_stream():
    pass  # Placeholder for future implementation


def load_data(subject=1, config=None):
    print(f"--- Loading Subject {subject} ---")
    # 3 Runs of the Same Session
    runs = [4, 8, 12]  # Left/Right/Rest Hand Motor Imagery
    raw_fnames = eegbci.load_data(subject, runs, verbose=False)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]

    # Concatenate the runs
    raw = concatenate_raws(raws, verbose=False)

    raw.rename_channels(lambda x: x.strip("."))  # remove dots from channel names
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))

    # Only predefined 16 channels out of 64 -> To be able to use with the dataset from BCI-Challenge
    raw = raw.copy().pick(config.channels)

    # Extract the labels with timings
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    return raw, events, event_id


def extract_baseline(subject=1):
    print(f"--- Loading Subject {subject} ---")
    runs = [1]  # Baseline with eyes open
    raw_fnames = eegbci.load_data(subject, runs, verbose=False)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw = raws[0]

    raw.rename_channels(lambda x: x.strip("."))  # remove dots from channel names

    eegbci.standardize(raw)

    raw.set_montage(make_standard_montage("standard_1005"))

    # Only predefined 16 channels out of 64
    raw = raw.copy().pick(config.channels)

    return raw

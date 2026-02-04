import mne
import pyxdf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


from bci.utils.bci_config import load_config


def _get_raw_xdf_offline(
    trial: Path,
    marker_durations: list[float] | None = None,
) -> tuple[mne.io.RawArray, list, list[str]]:
    """
    Load raw data from XDF, handle P999/P554 remapping, and set annotations.
    """
    print("\n")
    print(("=" * 30) + f" Processing file: {trial.name} " + ("=" * 30))
    print(f"Loading trial from: {trial}")

    if marker_durations is None:
        marker_durations = [3, 1, 3]

    streams, header = pyxdf.load_xdf(trial, verbose=False)

    event_channel = None
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

    # 1. Identify Streams
    for i, stream in enumerate(streams):
        name = stream["info"]["name"][0]
        if name == "EEG - Impedances":
            continue
        elif name in ["EEG", "EEG-GTEC"]:
            eeg_channel = i
            print("EEG stream found.")
        elif any(x in name for x in ["pupil_capture", "fixations"]):
            continue
        else:
            print(f"Unknown stream found (treating as markers): {name}")
            event_channel = i

    if eeg_channel is None:
        print("Error: No EEG stream found.")
        return None, None, None

    # 2. Extract Channel Labels
    if streams[eeg_channel]["info"]["desc"] != [None]:
        channel_dict = streams[eeg_channel]["info"]["desc"][0]["channels"][0]["channel"]
        channel_labels = [channel["label"][0] for channel in channel_dict]
    else:
        count = int(streams[eeg_channel]["info"]["channel_count"][0])
        channel_labels = standard_channels

    # 3. P999 / P554 Specific Remapping
    if "sub-P999_ses-S001_task-comp_final_run-001_eeg" in trial.name:
        print("Switched channels Cz and Fp2 detected. Reassigning labels...")
        channel_labels = [
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
    elif "P554" in trial.name:
        print("P554 recording detected. Assigning custom labels + keyboard...")
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
            "Keyboard",
        ]

    # 4. Prepare Data
    data = streams[eeg_channel]["time_series"].T * 1e-6  # scale to Volts

    # Strip non-EEG channels
    if channel_labels[-1].lower() in ["ts", "impedances", "keyboard"]:
        print(f"Discarding last channel: {channel_labels[-1]}")
        # data = data[:-1, :]
        channel_labels = channel_labels[:-1]

    if data.shape[0] != len(channel_labels):
        print(f"Mismatch: Data {data.shape[0]} ch vs Labels {len(channel_labels)}.")
        return None, None, None

    # 5. Create MNE Raw Object
    sfreq = float(streams[eeg_channel]["info"]["nominal_srate"][0])
    info = mne.create_info(channel_labels, sfreq, ch_types="eeg")
    raw_data = mne.io.RawArray(data, info, verbose=False)

    # 6. Apply Montage and Reorder
    montage = mne.channels.make_standard_montage("standard_1020")
    raw_data.set_montage(montage)

    if "P554" in trial.name or "sub-P999" in trial.name:
        print("Reordering channels to standard 10-20 montage...")
        raw_data.reorder_channels(standard_channels)
        channel_labels = raw_data.ch_names

    # 7. Handle Annotations
    markers = []
    if event_channel is not None:
        marker_names = np.array(streams[event_channel]["time_series"]).squeeze()
        time_marker = np.array(streams[event_channel]["time_stamps"]).squeeze()
        time_data = np.array(streams[eeg_channel]["time_stamps"])

        real_time_marker = (time_marker - time_data[0]).astype(float)
        duration_list = [
            marker_durations[i % len(marker_durations)]
            for i in range(len(real_time_marker))
        ]

        annotations = mne.Annotations(
            onset=real_time_marker, duration=3, description=marker_names
        )
        raw_data.set_annotations(annotations)
        markers = sorted(list(set(marker_names)))

    return raw_data, markers, channel_labels


def main():
    """Execution block to run the script."""
    # --- EDIT THIS PATH ---
    abs_path = r"/Users/iustincurcean/bci_practical/BCI-Challenge/data/eeg/sub/sub-P999_ses-S099_task-dino_run-002_eeg.xdf"
    # ----------------------

    trial_file = Path(abs_path)

    current_wd = Path.cwd()  # BCI-Challenge directory
    config_path = current_wd / "resources" / "configs" / "bci_config.yaml"
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    print("Configuration loaded successfully!")

    if not trial_file.exists():
        print(f"Error: File not found at {abs_path}")
        return

    # Load using your logic
    raw, markers, labels = _get_raw_xdf_offline(trial_file)

    from bci.preprocessing.filters import Filter

    filter_obj = Filter(config, online=False)

    # FILTERING: Apply bandpass filter
    filtered_raw = raw.copy()
    filtered_raw.apply_function(filter_obj.apply_filter_offline)

    # CHANNEL REMOVAL: Remove unnecessary channels (noise sources)
    # filtered_raw.drop_channels(config.remove_channels)

    if raw is not None:
        print(f"\nOpening Interactive Plot...")
        print(f"Channels: {labels}")
        print(f"Events: {markers}")

        # Launch MNE Browser
        # scalings='auto' helps since we multiplied by 1e-6
        filtered_raw.plot(
            block=True,
            n_channels=len(labels),
            title=f"Offline Inspection: {trial_file.name}",
            scalings={"eeg": 20e-5},
        )


if __name__ == "__main__":
    main()

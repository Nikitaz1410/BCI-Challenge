import sys
import time
import logging
import numpy as np
from pathlib import Path
from pylsl import StreamInfo, StreamOutlet

# --- Path Setup ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from bci.loading.loading import (
    create_subject_train_set,
    load_physionet_data,
    load_target_subject_data,
)
from bci.utils.bci_config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CMD_MAP = {0: "CIRCLE ONSET", 1: "ARROW LEFT ONSET", 2: "ARROW RIGHT ONSET"}


class BCIReplayer:
    def __init__(self, config):
        self.config = config
        self.fs = config.fs
        self.chunk_size = 32
        self._init_outlets()

    def _init_outlets(self):
        """Initialize LSL outlets."""
        # EEG Outlet
        info_eeg = StreamInfo(
            name="EEG_Stream",
            type="EEG",
            channel_count=len(self.config.channels),
            nominal_srate=self.fs,
            channel_format="float32",
            source_id="bci_replay_eeg_001",
        )
        self.outlet_eeg = StreamOutlet(info_eeg)

        # Labels Outlet (Sparse Markers)
        # nominal_srate=0 indicates an irregular/sparse stream
        info_labels = StreamInfo(
            name="Labels_Stream",
            type="Markers",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id="bci_replay_labels_001",
        )
        self.outlet_labels = StreamOutlet(info_labels)
        logger.info(
            f"LSL Streams initialized. EEG: {self.fs}Hz, Markers: Sparse (Event-based)."
        )

    def stream(self, eeg_data: np.ndarray, events: np.ndarray):
        """
        Stream EEG data continuously and inject Markers only when they occur.

        Parameters:
        - eeg_data: (n_channels, n_samples)
        - events: (n_events, 3) array [sample_index, 0, label_id]
        """
        n_channels, n_samples = eeg_data.shape
        chunk_duration = self.chunk_size / self.fs
        total_chunks = n_samples // self.chunk_size

        # Sort events by sample index to ensure efficient processing
        # (Though MNE usually provides them sorted)
        if len(events) > 0:
            events = events[np.argsort(events[:, 0])]

        logger.info(
            f"Starting Replay: {n_samples} samples, {len(events)} sparse events."
        )

        start_time = time.perf_counter()

        # Pointer to track which event we are currently waiting for
        event_ptr = 0
        n_events = len(events)

        for i in range(total_chunks):
            # 1. Define Current Window
            start_idx = i * self.chunk_size
            end_idx = start_idx + self.chunk_size

            # 2. Push EEG Chunk
            chunk_eeg = eeg_data[:, start_idx:end_idx]
            self.outlet_eeg.push_chunk(chunk_eeg.T.tolist())

            # 3. Check for Events in this Window (Synchronization)
            # We check if the next event in the list falls within [start_idx, end_idx)
            while event_ptr < n_events:
                ev_sample = events[event_ptr][0]
                ev_id = events[event_ptr][2]

                if start_idx <= ev_sample < end_idx:
                    # Event is inside this chunk! Push it now.
                    # Note: We send it immediately. Since we are in the loop processing
                    # this specific chunk, the arrival time at the receiver will be
                    # virtually identical to the EEG chunk arrival.
                    logger.info(f"Sending Marker: {ev_id} at sample {ev_sample}")
                    self.outlet_labels.push_sample([CMD_MAP[int(ev_id)]])

                    event_ptr += 1
                elif ev_sample < start_idx:
                    # Catchup: If we somehow skipped an event (shouldn't happen with correct logic)
                    event_ptr += 1
                else:
                    # Event is in the future (beyond end_idx), stop checking
                    break

            # 4. Drift Correction (Sleep)
            target_time = start_time + ((i + 1) * chunk_duration)
            current_time = time.perf_counter()
            sleep_duration = target_time - current_time

            if sleep_duration > 0:
                time.sleep(sleep_duration)

            if i % (int(self.fs / self.chunk_size) * 5) == 0:
                logger.debug(f"Stream heartbeat: {i}/{total_chunks}")

        logger.info("Replay finished.")


def load_dataset(config, current_wd):
    """Loads Raw MNE object and Events array."""
    logger.info("Loading Dataset...")

    if "Phy" in config.replay_subject_id:
        s_id = int(config.replay_subject_id.split("-")[1])
        raw, events, _, _, _ = load_physionet_data(
            subjects=[s_id], root=current_wd, channels=config.channels
        )
        # raw is a list of [Raw], events is list of [Arr]
        return raw[0], events[0]

    else:
        test_data_source = current_wd / "data" / "eeg" / config.target
        test_data_target = current_wd / "data" / "datasets" / config.target

        (all_raws, all_events, _, _, meta) = load_target_subject_data(
            root=current_wd,
            source_path=test_data_source,
            target_path=test_data_target,
            resample=None,
        )

        s_id = int(config.replay_subject_id.split("-")[1])

        # Load specific subject
        raws, events, _, _, _ = create_subject_train_set(
            config,
            all_raws,
            all_events,
            meta["filenames"],
            num_general=0,
            num_dino=s_id,
            num_supression=0,
            shuffle=False,
        )
        # raws is a list, events is a list of arrays
        return raws[s_id - 1], events[s_id - 1]


if __name__ == "__main__":
    # --- Configuration ---
    current_wd = Path(__file__).resolve().parent.parent.parent
    config_path = current_wd / "resources" / "configs" / "bci_config.yaml"

    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Could not load config: {e}")
        sys.exit(1)

    # --- Load Data ---
    try:
        mne_raw, mne_events = load_dataset(config, current_wd)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)

    # --- Pre-process ---
    eeg_data = mne_raw.get_data()  # (n_channels, n_samples)

    # Ensure events are standard numpy array
    if isinstance(mne_events, list):
        mne_events = np.array(mne_events)

    # --- Start Stream ---
    replayer = BCIReplayer(config)

    try:
        replayer.stream(eeg_data, mne_events)
    except KeyboardInterrupt:
        logger.info("Replay stopped by user.")

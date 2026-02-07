import sys
import time
import logging
import numpy as np
import scipy.signal
from pathlib import Path
from typing import Dict
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

# --- Configuration Toggle ---
synthetic = False  # True: Sine wave | False: Real EEG
synthetic_freq = 10.0  # Frequency in Hz
synthetic_amp = 50.0  # Amplitude (typical EEG scale in uV)


class BCIReplayer:
    def __init__(self, config):
        self.config = config
        self.fs = config.fs
        self.chunk_size = 32
        self._init_outlets()

    def _init_outlets(self):
        """Initialize LSL outlets."""
        info_eeg = StreamInfo(
            name="EEG_Stream",
            type="EEG",
            channel_count=len(self.config.channels),
            nominal_srate=self.fs,
            channel_format="float32",
            source_id="bci_replay_eeg_001",
        )
        self.outlet_eeg = StreamOutlet(info_eeg)

        info_labels = StreamInfo(
            name="Labels_Stream",
            type="Markers",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id="bci_replay_labels_001",
        )
        self.outlet_labels = StreamOutlet(info_labels)

        # Discoverability delay
        self.outlet_labels.push_sample(["STREAM_READY"])
        time.sleep(1)

    def stream(self, eeg_data: np.ndarray, events: np.ndarray):
        """
        Stream EEG or Synthetic data.
        """
        n_channels, n_samples = eeg_data.shape
        chunk_duration = self.chunk_size / self.fs
        total_chunks = n_samples // self.chunk_size

        if len(events) > 0:
            events = events[np.argsort(events[:, 0])]

        mode_str = f"SYNTHETIC ({synthetic_freq}Hz)" if synthetic else "REAL DATA"
        logger.info(f"Starting Replay [{mode_str}]: {n_samples} samples.")

        start_time = time.perf_counter()
        event_ptr = 0
        n_events = len(events)

        for i in range(total_chunks):
            start_idx = i * self.chunk_size
            end_idx = start_idx + self.chunk_size

            # --- Data Generation/Selection ---
            if synthetic:
                # Time vector for this chunk to maintain phase continuity
                t = np.arange(start_idx, end_idx) / self.fs
                sine_val = synthetic_amp * np.sin(2 * np.pi * synthetic_freq * t)
                chunk_eeg = np.tile(sine_val, (n_channels, 1)).astype(np.float32)
            else:
                chunk_eeg = eeg_data[:, start_idx:end_idx]

            # Push EEG
            self.outlet_eeg.push_chunk(chunk_eeg.T.tolist())

            # --- Marker Injection ---
            while event_ptr < n_events:
                ev_sample = events[event_ptr][0]
                ev_id = events[event_ptr][2]

                if start_idx <= ev_sample < end_idx:
                    marker_name = CMD_MAP.get(int(ev_id), "UNKNOWN")
                    logger.info(f"Marker: {marker_name} at sample {ev_sample}")
                    self.outlet_labels.push_sample([marker_name])
                    event_ptr += 1
                elif ev_sample < start_idx:
                    event_ptr += 1
                else:
                    break

            # Drift Correction
            target_time = start_time + ((i + 1) * chunk_duration)
            current_time = time.perf_counter()
            sleep_duration = target_time - current_time
            if sleep_duration > 0:
                time.sleep(sleep_duration)

        logger.info("Replay finished. Keep-alive active.")
        self._keep_alive(n_channels)

    def _keep_alive(self, n_channels):
        """Maintains streams after replay ends."""
        try:
            dummy_eeg = np.zeros((n_channels, self.chunk_size), dtype=np.float32)
            while True:
                time.sleep(0.5)
                self.outlet_eeg.push_chunk(dummy_eeg.T.tolist())
                self.outlet_labels.push_sample(["STREAM_ALIVE"])
        except KeyboardInterrupt:
            logger.info("Streaming stopped.")


def get_filter_metrics(sos, fs, passband) -> Dict[str, float]:
    """Calculates physical filter latency metrics."""
    # Compute group delay using sosfreqz for precision
    w, h = scipy.signal.sosfreqz(sos, worN=2000)
    # Note: sos2tf is used here only for analysis, not filtering
    _, gd = scipy.signal.group_delay(scipy.signal.sos2tf(sos), w)

    freqs_hz = w * fs / (2 * np.pi)
    mask = (freqs_hz >= passband[0]) & (freqs_hz <= passband[1])

    passband_gd = gd[mask]
    return {
        "avg_latency_ms": (np.mean(passband_gd) / fs) * 1000,
        "max_latency_ms": (np.max(passband_gd) / fs) * 1000,
    }


# --- Standard Loading Logic (Unchanged) ---
def load_dataset(config, current_wd):
    logger.info("Loading Dataset...")
    if "Phy" in config.replay_subject_id:
        s_id = int(config.replay_subject_id.split("-")[1])
        raw, events, _, _, _ = load_physionet_data(
            subjects=[s_id], root=current_wd, channels=config.channels
        )
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
        return raws[s_id - 1], events[s_id - 1]


if __name__ == "__main__":
    current_wd = Path(__file__).resolve().parent.parent.parent
    config_path = current_wd / "resources" / "configs" / "bci_config.yaml"

    try:
        config = load_config(config_path)
        mne_raw, mne_events = load_dataset(config, current_wd)

        # Timing Metrics for the Report
        # (Assuming you have a filter object/sos defined in your config)
        # metrics = get_filter_metrics(filter_obj.sos, config.fs, [8, 30])
        # logger.info(f"Filter Passband Latency: {metrics['avg_latency_ms']:.2f}ms (Avg)")

        replayer = BCIReplayer(config)
        replayer.stream(mne_raw.get_data(), np.array(mne_events))

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)

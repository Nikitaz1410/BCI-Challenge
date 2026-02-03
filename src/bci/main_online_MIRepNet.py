"""
Online BCI Pipeline for MIRepNet Foundation Model.

This script runs real-time EEG classification using a pre-trained MIRepNet model.
It mirrors the structure of main_online_Riemann.py and expects:

1. A pre-trained MIRepNet model saved by main_offline_MIRepNet.py
   (resources/models/mirepnet_best_model.pt)
   or by main_offline_MIRepNet_ar.py
   (resources/models/mirepnet_ar_best_model.pt)
2. Artifact rejection thresholds, e.g. for the _ar variant:
   (resources/models/mirepnet_artefact_removal.pkl)

Pipeline:
1. Load configuration and pre-trained MIRepNet model
2. Connect to LSL EEG and marker streams
3. Continuously: acquire → filter → artifact reject → classify → send to game

Usage:
    python main_online_MIRepNet.py
"""

import copy
import logging
import pickle
import socket
import sys
import time
from pathlib import Path

# Add src directory to Python path to allow imports
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import zmq
from pylsl import StreamInlet, resolve_streams

from bci.models.MIRepNet import MIRepNetModel
from bci.preprocessing.filters import Filter
from bci.transfer.transfer import BCIController
from bci.utils.bci_config import load_config

# --- Configuration & Constants ---
LOG_LEVEL = logging.INFO
ZMQ_PORT = "5556"
CMD_MAP = {"CIRCLE ONSET": 0, "ARROW LEFT ONSET": 1, "ARROW RIGHT ONSET": 2}
UNKNOWN = "UNKNOWN"

# Setup Logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BCIEngine:
    def __init__(self, config_path: Path):
        self.running = True
        self.stats = {
            "predictions": [0, 0, 0],
            "accepted": {
                "correct": [0, 0, 0],
                "incorrect": [0, 0, 0],
            },
            "rejected": {
                "correct": [0, 0, 0],
                "incorrect": [0, 0, 0],
            },
            "total_time_ms": 0.0,
        }

        # 1. Load Configuration
        try:
            logger.info(f"Loading configuration from: {config_path}")
            self.config = load_config(config_path)
            if hasattr(self.config, "random_state"):
                np.random.seed(self.config.random_state)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)

        # 2. Initialize Models & Filters
        self._init_models()

        # 3. Initialize Buffers
        self._init_buffers()

        # 4. Initialize Networking
        self._init_sockets()

    def _init_models(self):
        """Load MIRepNet model and signal filter."""
        base_path = Path.cwd() / "resources" / "models"
        # Prefer AR-trained MIRepNet model if available, otherwise fall back.
        ar_model_path = base_path / "mirepnet_ar_best_model.pt"
        default_model_path = base_path / "mirepnet_best_model.pt"
        model_path = ar_model_path if ar_model_path.exists() else default_model_path

        # MIRepNet-specific artefact removal thresholds (if available).
        artifact_path = base_path / "mirepnet_artefact_removal.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"MIRepNet model not found at {model_path}. "
                "Train the model first using main_offline_MIRepNet_ar.py "
                "or main_offline_MIRepNet.py"
            )

        # Signal Filter: MIRepNet uses all channels (no removal), so Filter must
        # initialize zi for len(channels) channels, not len(channels)-len(remove_channels)
        filter_config = copy.copy(self.config)
        filter_config.remove_channels = []
        self.signal_filter = Filter(filter_config, online=True)

        # MIRepNet Classifier (uses all channels, no removal, to match offline)
        logger.info(f"Loading MIRepNet model from {model_path}")
        self.clf = MIRepNetModel.load(str(model_path), device="auto")
        self._ensure_ea_ref_for_online()

        # Artifact Rejection: only use MIRepNet-specific AR if available.
        if artifact_path.exists():
            with open(artifact_path, "rb") as f:
                self.ar = pickle.load(f)
            logger.info(f"Loaded artifact rejection from {artifact_path}")
        else:
            logger.warning(
                "MIRepNet artefact rejection file not found. "
                "Running without artefact rejection."
            )
            self.ar = None

        # Controller (handles robust buffer logic)
        self.controller = BCIController(self.config)

    def _ensure_ea_ref_for_online(self):
        """
        Ensure EA reference for subject 0 exists for online prediction.

        The model may have been trained with different subject IDs (e.g. 554, 999).
        For online streaming we use subject_id=0, so we need a ref for key 0.
        """
        if self.clf._ea_ref_by_subject is None:
            self.clf._ea_ref_by_subject = {}
        refs = self.clf._ea_ref_by_subject
        if 0 not in refs and refs:
            first_ref = next(iter(refs.values()))
            refs[0] = first_ref
            logger.info("Set EA reference for subject 0 (online) from first available.")

    def _init_buffers(self):
        """Pre-allocate numpy buffers. MIRepNet uses all channels (no removal)."""
        # Match offline MIRepNet: use all channels, no channel removal
        self.keep_mask = np.arange(len(self.config.channels))

        n_channels = len(self.keep_mask)
        window_size = int(self.config.window_size)

        logger.info(
            f"Initialized Buffer: {n_channels} channels x {window_size} samples"
        )

        self.buffer = np.zeros((n_channels, window_size), dtype=np.float32)

    def _init_sockets(self):
        """Setup ZMQ and UDP sockets."""
        self.zmq_ctx = zmq.Context()
        self.zmq_pub = self.zmq_ctx.socket(zmq.PUB)
        self.zmq_pub.bind(f"tcp://*:{ZMQ_PORT}")
        logger.info(f"ZMQ Publisher bound to port {ZMQ_PORT}")

        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _connect_streams(self):
        """Resolve and connect to LSL streams."""
        logger.info("Resolving LSL streams...")
        streams = resolve_streams(wait_time=5.0)

        eeg_streams = [s for s in streams if s.type() == "EEG"]

        target_label_stream = (
            "MyDinoGameMarkerStream"
            if self.config.online == "dino"
            else "Labels_Stream"
        )
        label_streams = [
            s
            for s in streams
            if s.type() == "Markers" and s.name() == target_label_stream
        ]

        if not eeg_streams:
            raise RuntimeError("No EEG Stream found.")

        if label_streams:
            self.inlet_labels = StreamInlet(label_streams[0], max_chunklen=32)
            logger.info(f"Connected to Markers: {label_streams[0].name()}")
        else:
            logger.warning(
                "No Marker Stream found. Operating without ground truth labels."
            )
            self.inlet_labels = None

        self.inlet_eeg = StreamInlet(eeg_streams[0], max_chunklen=32)
        logger.info(f"Connected to EEG: {eeg_streams[0].name()}")

    def _update_buffer(self, buffer: np.ndarray, new_data: np.ndarray) -> np.ndarray:
        """Efficiently shift buffer and append new data."""
        n_new = new_data.shape[1]
        if n_new == 0:
            return buffer

        if n_new >= buffer.shape[1]:
            return new_data[:, -buffer.shape[1] :].copy()
        buffer[:, :-n_new] = buffer[:, n_new:]
        buffer[:, -n_new:] = new_data
        return buffer

    def run(self):
        try:
            self._connect_streams()
        except RuntimeError as e:
            logger.error(e)
            return

        logger.info("Starting BCI Loop (MIRepNet)...")

        crt_label = UNKNOWN

        try:
            while self.running:
                start_time = time.perf_counter()

                # --- 1. Data Acquisition ---
                chunk, timestamp = self.inlet_eeg.pull_chunk()

                if self.inlet_labels:
                    label, timestamp = self.inlet_labels.pull_chunk()
                    if label:
                        crt_label = label[0][0]
                        if crt_label in CMD_MAP.keys():
                            logger.info(f"{crt_label}: {CMD_MAP[crt_label]}")
                        else:
                            crt_label = UNKNOWN

                if not chunk:
                    continue

                # --- 2. Preprocessing ---
                eeg_np = np.array(chunk).T[self.keep_mask]
                filtered_chunk = self.signal_filter.apply_filter_online(eeg_np)

                self.buffer = self._update_buffer(self.buffer, filtered_chunk)

                self.zmq_pub.send_pyobj(("DATA", eeg_np, filtered_chunk))

                # --- 3. Artifact Rejection ---
                if self.ar is not None and self.ar.reject_bad_epochs_online(
                    self.buffer
                ):
                    self.zmq_pub.send_pyobj(("EVENT", "Artifact"))
                    continue

                # --- 4. Classification (MIRepNet expects shape [N, C, T]) ---
                window = self.buffer[np.newaxis, :, :].astype(np.float32)
                probs = self.clf.predict_proba(window, subject_ids=np.zeros(1, dtype=np.int64))
                probs = np.asarray(probs)
                if probs.size == 0:
                    continue
                # BCIController expects (1, n_classes) so .T yields (n_classes, 1) for assignment
                probs = probs.reshape(1, -1)
                prediction = int(np.argmax(probs))

                # --- 5. Robust Buffer Logic ---
                buffer_proba = self.controller.send_command(probs, self.udp_sock)

                if buffer_proba is not None:
                    self.zmq_pub.send_pyobj(("PROBA", buffer_proba.flatten()))

                    # --- 6. Evaluation ---
                    if crt_label != UNKNOWN:
                        ground_truth = CMD_MAP[crt_label]

                        self.stats["predictions"][ground_truth] += 1

                        if np.max(buffer_proba) < self.config.classification_threshold:
                            if ground_truth == prediction:
                                self.stats["rejected"]["correct"][ground_truth] += 1
                            else:
                                self.stats["rejected"]["incorrect"][ground_truth] += 1
                        else:
                            if ground_truth == prediction:
                                self.stats["accepted"]["correct"][ground_truth] += 1
                            else:
                                self.stats["accepted"]["incorrect"][ground_truth] += 1

                dt_ms = (time.perf_counter() - start_time) * 1000
                self.stats["total_time_ms"] += dt_ms

        except KeyboardInterrupt:
            self.stop()

    def _print_class_stats(self):
        """Print formatted table of class-wise statistics."""
        class_names = ["Rest", "Left", "Right"]

        print("\n" + "=" * 85)
        print(f"{'CLASS PERFORMANCE REPORT (MIRepNet)':^85}")
        print("=" * 85)

        header = (
            f"| {'Class':<12} | {'Total':<6} | "
            f"{'Accepted (Corr/Inc)':<20} | "
            f"{'Rejected':<10} | "
            f"{'Rej Rate':<10} | "
            f"{'Accuracy':<10} |"
        )
        print(header)
        print("-" * 85)

        total_correct_accepted = 0
        total_accepted = 0
        total_predictions = 0

        for i, name in enumerate(class_names):
            n_total = self.stats["predictions"][i]

            acc_corr = self.stats["accepted"]["correct"][i]
            acc_inc = self.stats["accepted"]["incorrect"][i]
            n_accepted = acc_corr + acc_inc

            rej_corr = self.stats["rejected"]["correct"][i]
            rej_inc = self.stats["rejected"]["incorrect"][i]
            n_rejected = rej_corr + rej_inc

            total_predictions += n_total
            total_correct_accepted += acc_corr
            total_accepted += n_accepted

            if n_accepted > 0:
                accuracy = (acc_corr / n_accepted) * 100
            else:
                accuracy = 0.0

            if n_total > 0:
                rej_rate = (n_rejected / n_total) * 100
            else:
                rej_rate = 0.0

            print(
                f"| {name:<12} | {n_total:<6} | "
                f"{f'{acc_corr}/{acc_inc}':<20} | "
                f"{n_rejected:<10} | "
                f"{rej_rate:<9.1f}% | "
                f"{accuracy:<9.1f}% |"
            )

        print("-" * 85)

        if total_accepted > 0:
            global_acc = (total_correct_accepted / total_accepted) * 100
        else:
            global_acc = 0.0

        if total_predictions > 0:
            global_rej = (
                (total_predictions - total_accepted) / total_predictions
            ) * 100
        else:
            global_rej = 0.0

        print(
            f"| {'OVERALL':<12} | {total_predictions:<6} | "
            f"{'':<20} | "
            f"{total_predictions - total_accepted:<10} | "
            f"{global_rej:<9.1f}% | "
            f"{global_acc:<9.1f}% |"
        )
        print("=" * 85 + "\n")

    def stop(self):
        """Graceful shutdown."""
        self.running = False
        logger.info("Stopping BCI Engine (MIRepNet).")

        n_preds = max(1, sum(self.stats["predictions"]))
        avg_time = self.stats["total_time_ms"] / n_preds

        self._print_class_stats()

        print(f"Avg Processing Time per Epoch: {avg_time:.2f} ms")
        print("=" * 30 + "\n")

        self.udp_sock.close()
        self.zmq_ctx.destroy()


if __name__ == "__main__":
    current_wd = Path.cwd()
    conf_path = current_wd / "resources" / "configs" / "bci_config.yaml"

    engine = BCIEngine(conf_path)
    engine.run()

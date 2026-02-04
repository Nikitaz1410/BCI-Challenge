"""
Online BCI Pipeline for the MIRepNet Foundation Model.

This script implements a real-time BCI system using the MIRepNet deep learning
model for motor imagery classification. It follows the same pattern as
main_online_Riemann.py but uses the neural network approach.

Pipeline:
1. Load configuration and pre-trained MIRepNet model
2. Connect to LSL streams (EEG data and optional markers)
3. Real-time loop:
   - Acquire EEG data
   - Apply bandpass filtering
   - Run artifact rejection
   - Classify with MIRepNet (model handles EA and channel interpolation)
   - Apply robust buffering for stable predictions
   - Send commands via UDP (for game control)
   - Publish data via ZMQ (for visualization)

Requirements:
- Pre-trained MIRepNet model with Autoreject (from main_offline_MIRepNet.py)
- LSL EEG stream
- Optional: LSL Markers stream for evaluation

Usage:
    python main_online_MIRepNet_adapt.py
"""

import logging
import pickle
import socket
import sys
import time
from pathlib import Path

# Add src directory to Python path to allow imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import torch
import zmq
from pylsl import StreamInlet, resolve_streams

# Local imports
from bci.models.MIRepNet import MIRepNetModel
from bci.preprocessing.filters import Filter
from bci.transfer.transfer import BCIController
from bci.utils.bci_config import load_config

# --- Configuration & Constants ---
LOG_LEVEL = logging.INFO
ZMQ_PORT = "5556"  
# Maps string commands to integer classes
CMD_MAP = {"CIRCLE ONSET": 0, "ARROW LEFT ONSET": 1, "ARROW RIGHT ONSET": 2}
UNKNOWN = "UNKNOWN"

# Setup Logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class MIRepNetBCIEngine:
    """
    Real-time BCI engine using MIRepNet for motor imagery classification.

    This engine manages:
    - LSL stream connections for EEG data and markers
    - Real-time signal filtering
    - Artifact rejection
    - MIRepNet inference with proper preprocessing
    - Robust prediction buffering
    - Output to game (UDP) and visualizer (ZMQ)
    """

    def __init__(self, config_path: Path):
        """
        Initialize the MIRepNet BCI Engine.

        Parameters
        ----------
        config_path : Path
            Path to the BCI configuration YAML file.
        """
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
            "inference_time_ms": [],
        }

        # 1. Load Configuration
        try:
            logger.info(f"Loading configuration from: {config_path}")
            self.config = load_config(config_path)
            # Set random seed
            if hasattr(self.config, "random_state"):
                np.random.seed(self.config.random_state)
                torch.manual_seed(self.config.random_state)
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
        """Load MIRepNet model, artifact rejection, and signal filters."""
        base_path = Path.cwd() / "resources" / "models"
        # Model and artefact removal trained in main_offline_MIRepNet.py
        model_path = base_path / "mirepnet_ar_best_model.pt"
        artifact_path = base_path / "mirepnet_artefact_removal.pkl"
        self.ea_adapted_path = base_path / "mirepnet_online_ea_adapted.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"MIRepNet model not found at {model_path}. "
                "Train and save the model first using main_offline_MIRepNet_ar.py"
            )

        # Signal Filter (online mode; use all channels)
        self.signal_filter = Filter(
            self.config, online=True, n_channels_online=len(self.config.channels)
        )
        logger.info("Signal filter initialized (online mode)")

        logger.info(f"Loading MIRepNet model from: {model_path}")
        self.clf = MIRepNetModel.load(str(model_path))
        self.clf.actual_channels = list(self.config.channels)
        logger.info(f"MIRepNet model loaded successfully (device: {self.clf.device})")
        logger.info(f"Model configuration: {self.clf._n_classes} classes")

        # Initialize online EA adaptation (warm start from saved state if exists)
        self.clf.init_online_ea(
            alpha=0.2, min_samples=100, saved_path=str(self.ea_adapted_path)
        )
        logger.info("Online EA adaptation enabled (alpha=0.2, min_samples=100)")

        # Artifact Rejection (thresholds trained in main_offline_MIRepNet.py)
        if artifact_path.exists():
            with open(artifact_path, "rb") as f:
                self.ar = pickle.load(f)
            logger.info(f"MIRepNet ArtefactRemoval loaded from {artifact_path}")
            self.use_artifact_rejection = True
        else:
            logger.warning(
                "MIRepNet artefact rejection file not found. Running without artefact rejection."
            )
            self.ar = None
            self.use_artifact_rejection = False

        # Controller (handles robust buffer logic for stable predictions)
        self.controller = BCIController(self.config)
        logger.info("BCI Controller initialized")

    def _init_buffers(self):
        """Pre-allocate numpy buffers for real-time processing."""
        # Use all available channels (do not remove any)
        self.keep_mask = np.arange(len(self.config.channels))
        n_channels = len(self.keep_mask)
        window_size = int(self.config.window_size)

        logger.info(
            f"Buffer initialized: {n_channels} channels x {window_size} samples"
        )

        # Main EEG Data Buffer (sliding window)
        self.buffer = np.zeros((n_channels, window_size), dtype=np.float32)

    def _init_sockets(self):
        """Setup ZMQ and UDP sockets for output."""
        # ZMQ Publisher (for Visualizer)
        self.zmq_ctx = zmq.Context()
        self.zmq_pub = self.zmq_ctx.socket(zmq.PUB)
        self.zmq_pub.bind(f"tcp://*:{ZMQ_PORT}")
        logger.info(f"ZMQ Publisher bound to port {ZMQ_PORT}")

        # UDP Socket (for Game/External App)
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info("UDP socket initialized")

    def _connect_streams(self):
        """Resolve and connect to LSL streams."""
        logger.info("Resolving LSL streams...")
        streams = resolve_streams(wait_time=5.0)

        # Find EEG stream
        eeg_streams = [s for s in streams if s.type() == "EEG"]

        # Determine label stream name based on mode
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
        """Efficiently shift buffer and append new data (sliding window)."""
        n_new = new_data.shape[1]
        if n_new == 0:
            return buffer

        if n_new >= buffer.shape[1]:
            return new_data[:, -buffer.shape[1]:]
        else:
            buffer[:, :-n_new] = buffer[:, n_new:]
            buffer[:, -n_new:] = new_data
            return buffer

    def _classify(self, window: np.ndarray) -> tuple:
        """
        Run MIRepNet classification with online EA adaptation.

        Parameters
        ----------
        window : np.ndarray
            EEG window of shape (1, n_channels, n_samples)

        Returns
        -------
        tuple
            (probabilities, prediction)
        """
        inference_start = time.perf_counter()

        # Use adapt_ea=True for continuous domain adaptation
        probs = self.clf.predict_proba(window, adapt_ea=True)
        probs = probs.flatten()
        prediction = int(np.argmax(probs))

        inference_time = (time.perf_counter() - inference_start) * 1000
        self.stats["inference_time_ms"].append(inference_time)

        return probs, prediction

    def run(self):
        """Main BCI loop - acquires data, processes, classifies, and outputs."""
        try:
            self._connect_streams()
        except RuntimeError as e:
            logger.error(e)
            return

        logger.info("Starting MIRepNet BCI Loop...")
        logger.info(f"Classification threshold: {self.config.classification_threshold}")

        # Current label from marker stream (for evaluation)
        crt_label = UNKNOWN

        try:
            while self.running:
                start_time = time.perf_counter()

                # --- 1. Data Acquisition ---
                chunk, timestamp = self.inlet_eeg.pull_chunk()

                # Check for label updates
                if self.inlet_labels:
                    label, _ = self.inlet_labels.pull_chunk()
                    if label:
                        crt_label = label[0][0]
                        if crt_label in CMD_MAP.keys():
                            logger.info(f"{crt_label}: {CMD_MAP[crt_label]}")
                        else:
                            crt_label = UNKNOWN

                if not chunk:
                    continue

                # --- 2. Preprocessing ---
                # Transpose to (channels, samples) and select channels
                eeg_np = np.array(chunk).T[self.keep_mask]
                # Scale from microVolts (µV) to Volts (V) - classifier trained on V
                eeg_np = eeg_np * 1e-6  # Convert µV to V (multiply by 10^-6)
                filtered_chunk = self.signal_filter.apply_filter_online(eeg_np)

                # Update sliding window buffer
                self.buffer = self._update_buffer(self.buffer, filtered_chunk)

                # Send raw data to visualizer
                self.zmq_pub.send_pyobj(("DATA", eeg_np, filtered_chunk))

                # --- 3. Artifact Rejection ---
                if self.use_artifact_rejection and self.ar is not None:
                    if self.ar.reject_bad_epochs_online(self.buffer):
                        self.zmq_pub.send_pyobj(("EVENT", "Artifact"))
                        continue

                # --- 4. Classification ---
                window = self.buffer[np.newaxis, :, :].astype(np.float32)
                probs, prediction = self._classify(window)

                if probs is None:
                    continue

                # --- 5. Robust Buffer Logic ---
                # Send probabilities to controller for smoothing/buffering
                buffer_proba = self.controller.send_command(
                    probs.reshape(1, -1), self.udp_sock
                )

                # Send robust probabilities to visualizer
                if buffer_proba is not None:
                    self.zmq_pub.send_pyobj(("PROBA", buffer_proba.flatten()))

                    # --- 6. Evaluation (if ground truth available) ---
                    if crt_label != UNKNOWN:
                        ground_truth = CMD_MAP[crt_label]
                        self.stats["predictions"][ground_truth] += 1

                        # Check accepted vs rejected and correct vs incorrect
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

                # Timing
                dt_ms = (time.perf_counter() - start_time) * 1000
                self.stats["total_time_ms"] += dt_ms

        except KeyboardInterrupt:
            self.stop()

    def _print_class_stats(self):
        """Print formatted table of class-wise performance statistics."""
        class_names = ["Rest", "Left", "Right"]

        print("\n" + "=" * 85)
        print(f"{'MIREPNET CLASS PERFORMANCE REPORT':^85}")
        print("=" * 85)

        # Header
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

            # Accuracy is based only on ACCEPTED trials (Online Accuracy)
            accuracy = (acc_corr / n_accepted * 100) if n_accepted > 0 else 0.0
            rej_rate = (n_rejected / n_total * 100) if n_total > 0 else 0.0

            print(
                f"| {name:<12} | {n_total:<6} | "
                f"{f'{acc_corr}/{acc_inc}':<20} | "
                f"{n_rejected:<10} | "
                f"{rej_rate:<9.1f}% | "
                f"{accuracy:<9.1f}% |"
            )

        print("-" * 85)

        # Global summaries
        global_acc = (
            (total_correct_accepted / total_accepted * 100) if total_accepted > 0 else 0.0
        )
        global_rej = (
            ((total_predictions - total_accepted) / total_predictions * 100)
            if total_predictions > 0
            else 0.0
        )

        print(
            f"| {'OVERALL':<12} | {total_predictions:<6} | "
            f"{'':<20} | "
            f"{total_predictions - total_accepted:<10} | "
            f"{global_rej:<9.1f}% | "
            f"{global_acc:<9.1f}% |"
        )
        print("=" * 85 + "\n")

    def stop(self):
        """Graceful shutdown with statistics summary and EA state persistence."""
        self.running = False
        logger.info("Stopping MIRepNet BCI Engine.")

        # Save adapted EA for warm start on next run (same subject)
        if hasattr(self, "ea_adapted_path") and self.clf.save_online_ea(
            str(self.ea_adapted_path)
        ):
            logger.info(f"Saved adapted EA state to {self.ea_adapted_path}")

        # Calculate timing statistics
        n_preds = max(1, sum(self.stats["predictions"]))
        avg_time = self.stats["total_time_ms"] / n_preds

        # Print detailed performance table
        self._print_class_stats()

        # Inference timing
        if self.stats["inference_time_ms"]:
            avg_inference = np.mean(self.stats["inference_time_ms"])
            std_inference = np.std(self.stats["inference_time_ms"])
            print(f"Avg Inference Time: {avg_inference:.2f} +/- {std_inference:.2f} ms")

        print(f"Avg Total Processing Time per Window: {avg_time:.2f} ms")
        print("=" * 30 + "\n")

        # Cleanup
        self.udp_sock.close()
        self.zmq_ctx.destroy()


def main():
    """Entry point for online MIRepNet BCI."""
    current_wd = Path.cwd()
    conf_path = current_wd / "resources" / "configs" / "bci_config.yaml"

    engine = MIRepNetBCIEngine(conf_path)
    engine.run()


if __name__ == "__main__":
    main()

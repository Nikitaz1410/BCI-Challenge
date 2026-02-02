import logging
import pickle
import socket
import sys
import time
from pathlib import Path

import numpy as np
import zmq
from pylsl import StreamInlet, resolve_streams

# Local imports (ensure your python path is set correctly to find these)
from bci.models.riemann import RiemannianClf
from bci.preprocessing.filters import Filter
from bci.transfer.transfer import BCIController
from bci.utils.bci_config import load_config

# --- Configuration & Constants ---
LOG_LEVEL = logging.INFO
ZMQ_PORT = "5556"
# Maps string commands to integer classes if needed
CMD_MAP = {"CIRCLE ONSET": 0, "ARROW LEFT ONSET": 1, "ARROW RIGHT ONSET": 2}

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
            "predictions": 0,
            "rejected": 0,
            "successes": 0,
            "fails": 0,
            "total_time_ms": 0.0,
        }

        # 1. Load Configuration
        try:
            logger.info(f"Loading configuration from: {config_path}")
            self.config = load_config(config_path)
            # Set random seed
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
        """Load ML models and signal filters."""
        base_path = Path.cwd() / "resources" / "models"
        model_path = base_path / "model.pkl"
        artifact_path = base_path / "artefact_removal.pkl"

        # Signal Filter
        self.signal_filter = Filter(self.config, online=True)

        # Classifier
        model_args = {"cov_est": "lwf"}
        self.clf = RiemannianClf(model_args).load(model_path)

        # Artifact Rejection
        with open(artifact_path, "rb") as f:
            self.ar = pickle.load(f)

        # Controller (handles the robust buffer logic)
        self.controller = BCIController(self.config)

    def _init_buffers(self):
        """Pre-allocate numpy buffers."""
        self.keep_mask = np.array(
            [
                self.config.channels.index(ch)
                for ch in self.config.channels
                if ch not in self.config.remove_channels
            ]
        )
        n_channels = len(self.keep_mask)
        window_size = int(self.config.window_size)

        logger.info(
            f"Initialized Buffer: {n_channels} channels x {window_size} samples"
        )

        # Main Data Buffer
        self.buffer = np.zeros((n_channels, window_size), dtype=np.float32)

    def _init_sockets(self):
        """Setup ZMQ and UDP sockets."""
        # ZMQ Publisher (for Visualizer)
        self.zmq_ctx = zmq.Context()
        self.zmq_pub = self.zmq_ctx.socket(zmq.PUB)
        self.zmq_pub.bind(f"tcp://*:{ZMQ_PORT}")
        logger.info(f"ZMQ Publisher bound to port {ZMQ_PORT}")

        # UDP Socket (for Game/External App)
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _connect_streams(self):
        """Resolve and connect to LSL streams."""
        logger.info("Resolving LSL streams...")
        streams = resolve_streams(wait_time=5.0)

        eeg_streams = [s for s in streams if s.type() == "EEG"]

        # Determine Label Stream Name based on mode
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

    def _update_buffer(self, buffer, new_data):
        """Efficiently shifts buffer and appends new data."""
        n_new = new_data.shape[1]
        if n_new == 0:
            return buffer

        if n_new >= buffer.shape[1]:
            return new_data[:, -buffer.shape[1] :]
        else:
            buffer[:, :-n_new] = buffer[:, n_new:]
            buffer[:, -n_new:] = new_data
            return buffer

    def run(self):
        try:
            self._connect_streams()
        except RuntimeError as e:
            logger.error(e)
            return

        logger.info("Starting BCI Loop...")

        try:
            while self.running:
                start_time = time.perf_counter()

                # --- 1. Data Acquisition ---
                chunk, timestamp = self.inlet_eeg.pull_chunk()

                # Consume labels to keep stream clear (logic omitted for brevity if not used for calibration)
                if self.inlet_labels:
                    _ = self.inlet_labels.pull_chunk()

                if not chunk:
                    # Small sleep to prevent CPU spinning if LSL is non-blocking
                    # time.sleep(0.001)
                    continue

                # --- 2. Preprocessing ---
                # Transpose to (channels, samples) and filter
                eeg_np = np.array(chunk).T[self.keep_mask]
                filtered_chunk = self.signal_filter.apply_filter_online(eeg_np)

                # Update Main Buffer
                self.buffer = self._update_buffer(self.buffer, filtered_chunk)

                # Send Data to Visualizer
                self.zmq_pub.send_pyobj(("DATA", eeg_np, filtered_chunk))

                # --- 3. Artifact Rejection ---
                if self.ar.is_artifact(self.buffer):
                    # logger.debug("Artifact detected.")
                    self.zmq_pub.send_pyobj(("EVENT", "Artifact"))
                    continue

                # --- 4. Classification ---
                # Predict probabilities
                probs = self.clf.predict_proba(self.buffer)

                # Adapt the classifier if it is riemann
                if isinstance(self.clf, RiemannianClf):
                    self.clf.update_centroids()

                if probs is None:
                    continue

                # --- 5. Robust Buffer Logic ---
                # Send raw probs to controller, get back the smoothed/robust probabilities
                buffer_proba = self.controller.send_command(probs, self.udp_sock)

                # Send Robust Probabilities to Visualizer
                if buffer_proba is not None:
                    # Flatten to ensure it's a simple 1D array [p_rest, p_left, p_right]
                    self.zmq_pub.send_pyobj(("PROBA", buffer_proba.flatten()))

                # Stats & Logging
                self.stats["predictions"] += 1

                # Timing
                dt_ms = (time.perf_counter() - start_time) * 1000
                self.stats["total_time_ms"] += dt_ms

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Graceful shutdown."""
        self.running = False
        logger.info("Stopping BCI Engine.")

        n_preds = max(1, self.stats["predictions"])
        avg_time = self.stats["total_time_ms"] / n_preds

        print("\n" + "=" * 30)
        print(f"SESSION SUMMARY")
        print(f"Total Predictions: {self.stats['predictions']}")
        print(f"Avg Processing Time: {avg_time:.2f} ms")
        print("=" * 30 + "\n")

        self.udp_sock.close()
        self.zmq_ctx.destroy()


if __name__ == "__main__":
    current_wd = Path.cwd()
    conf_path = current_wd / "resources" / "configs" / "bci_config.yaml"

    engine = BCIEngine(conf_path)
    engine.run()

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
        # Timings
        self.filtering_time = 0.0
        self.ar_time = 0.0
        self.nr_of_loops = 0

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
        artifact_path = base_path / "ar.pkl"

        # Signal Filter
        self.signal_filter = Filter(self.config, online=True)

        # Classifier
        self.clf = RiemannianClf(cov_est="lwf").load(model_path)

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

        # self.inlet_eeg = StreamInlet(eeg_streams[0], max_chunklen=32)
        self.inlet_eeg = StreamInlet(eeg_streams[0])
        logger.info(
            f"Connected to EEG: {eeg_streams[0].name()} with ID {eeg_streams[0].source_id()}"
        )

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

        # Initialize the current label -> Is updated based on the MarkerStream
        crt_label = UNKNOWN

        try:
            while self.running:
                start_time = time.perf_counter()

                # --- 1. Data Acquisition ---
                chunk, timestamp = self.inlet_eeg.pull_chunk()

                # Consume labels to keep stream clear (logic omitted for brevity if not used for calibration)
                if self.inlet_labels:
                    label, timestamp = self.inlet_labels.pull_chunk()
                    if label:
                        crt_label = label[0][0]
                        if crt_label in CMD_MAP.keys():
                            logger.info(f"{crt_label}: {CMD_MAP[crt_label]}")
                        else:
                            crt_label = UNKNOWN

                if not chunk:
                    # Small sleep to prevent CPU spinning if LSL is non-blocking
                    # time.sleep(0.001)
                    continue

                # --- 2. Preprocessing ---
                # Transpose to (channels, samples) and filter
                eeg_np = np.array(chunk).T[self.keep_mask]  # Scale to microvolt
                if self.config.online == "dino":
                    eeg_np = eeg_np * 1e-6  # scale to microvolt
                start_f = time.perf_counter()
                filtered_chunk = self.signal_filter.apply_filter_online(eeg_np)
                self.filtering_time += (time.perf_counter() - start_f) * 1000

                # Update Main Buffer
                self.buffer = self._update_buffer(self.buffer, filtered_chunk)

                # Send Data to Visualizer
                self.zmq_pub.send_pyobj(("DATA", eeg_np, filtered_chunk))

                start_ar = time.perf_counter()
                # --- 3. Artifact Rejection ---
                is_artefact = self.ar.reject_bad_epochs_online(self.buffer)

                self.ar_time += (time.perf_counter() - start_ar) * 1000
                self.nr_of_loops += 1

                if is_artefact:
                    # logger.debug("Artifact detected.")
                    self.zmq_pub.send_pyobj(("EVENT", "Artifact"))
                    continue

                # --- 4. Classification ---
                # Predict probabilities
                # probs, cov = self.clf.predict_proba(self.buffer)
                # prediction = np.argmax(probs)
                prediction, probs, cov = self.clf.predict_with_recentering(
                    self.buffer
                )  # For normalized riemann predictor
                if prediction is None:
                    continue
                prediction = prediction[0]

                # Adapt the classifier if it is riemann
                if isinstance(self.clf, RiemannianClf):
                    # self.clf.adapt(
                    #    cov, prediction
                    # )  # UNCOMMENT FOR REALTIME PREDICTION-BASED ADAPTATION
                    pass

                if probs is None:
                    continue

                # --- 5. Robust Buffer Logic ---
                # Send raw probs to controller, get back the smoothed/robust probabilities
                buffer_proba = self.controller.send_command(probs, self.udp_sock)

                # Send Robust Probabilities to Visualizer
                if buffer_proba is not None:
                    # Flatten to ensure it's a simple 1D array [p_rest, p_left, p_right]
                    self.zmq_pub.send_pyobj(("PROBA", buffer_proba.flatten()))

                    # --- 6. Evaluation --- #
                    if crt_label != UNKNOWN:
                        ground_truth = CMD_MAP[crt_label]

                        # Stats & Logging
                        self.stats["predictions"][ground_truth] += 1

                        # Check the correct classifications and the accepted classifications
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
        """Helper to print a formatted table of class-wise statistics."""
        # Inverse the map to get names: {0: "CIRCLE...", 1: "LEFT..."}
        # Simplify names for the table

        # You might want to customize these short names based on your CMD_MAP
        # Assuming 0: Rest/Circle, 1: Left, 2: Right
        class_names = ["Rest", "Left", "Right"]

        print("\n" + "=" * 85)
        print(f"{'CLASS PERFORMANCE REPORT':^85}")
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

        # Iterate through classes (assuming indices 0, 1, 2)
        for i, name in enumerate(class_names):
            # Fetch stats
            n_total = self.stats["predictions"][i]

            acc_corr = self.stats["accepted"]["correct"][i]
            acc_inc = self.stats["accepted"]["incorrect"][i]
            n_accepted = acc_corr + acc_inc

            rej_corr = self.stats["rejected"]["correct"][i]
            rej_inc = self.stats["rejected"]["incorrect"][i]
            n_rejected = rej_corr + rej_inc

            # Global Accumulators
            total_predictions += n_total
            total_correct_accepted += acc_corr
            total_accepted += n_accepted

            # Metrics
            # Accuracy is based only on ACCEPTED trials (Online Accuracy)
            if n_accepted > 0:
                accuracy = (acc_corr / n_accepted) * 100
            else:
                accuracy = 0.0

            if n_total > 0:
                rej_rate = (n_rejected / n_total) * 100
            else:
                rej_rate = 0.0

            # Row Printing
            print(
                f"| {name:<12} | {n_total:<6} | "
                f"{f'{acc_corr}/{acc_inc}':<20} | "
                f"{n_rejected:<10} | "
                f"{rej_rate:<9.1f}% | "
                f"{accuracy:<9.1f}% |"
            )

        print("-" * 85)

        # Global Summaries
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

        # Footer Stats
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
        logger.info("Stopping BCI Engine.")

        # Calculate Timing
        n_preds = max(1, sum(self.stats["predictions"]))
        avg_time = self.stats["total_time_ms"] / n_preds

        # Print the detailed table
        self._print_class_stats()

        print(f"Avg Processing Time per Epoch: {avg_time:.2f} ms")
        print("=" * 30 + "\n")
        avg_fil = self.filtering_time / self.nr_of_loops
        avg_ar = self.ar_time / self.nr_of_loops
        print(f"Average filtering time: {avg_fil:.2f}")
        print(f"Average artefact check time: {avg_ar:.2f}")

        self.udp_sock.close()
        self.zmq_ctx.destroy()


if __name__ == "__main__":
    current_wd = Path.cwd()
    conf_path = current_wd / "resources" / "configs" / "bci_config.yaml"

    engine = BCIEngine(conf_path)
    engine.run()

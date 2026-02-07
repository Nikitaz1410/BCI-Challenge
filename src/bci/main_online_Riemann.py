import logging
import pickle
import socket
import sys
import time
from pathlib import Path

import numpy as np
import zmq
from pylsl import StreamInlet, resolve_streams

# Local imports
from bci.models.riemann import RiemannianClf
from bci.preprocessing.filters import Filter
from bci.transfer.transfer import BCIController
from bci.utils.bci_config import load_config

# --- Constants & Global Logging ---
ZMQ_PORT = "5556"
CMD_MAP = {"CIRCLE ONSET": 0, "ARROW LEFT ONSET": 1, "ARROW RIGHT ONSET": 2}
UNKNOWN = "UNKNOWN"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BCIEngine:
    """
    The central BCI Engine responsible for the real-time processing loop.

    It coordinates LSL stream acquisition, online filtering, artifact rejection,
    Riemannian classification, and external command dispatching.
    """

    def __init__(self, config_path: Path):
        self.running = True
        self.nr_of_loops = 0

        # Statistics Tracking
        self.stats = {
            "predictions": [0, 0, 0],
            "accepted": {"correct": [0, 0, 0], "incorrect": [0, 0, 0]},
            "rejected": {"correct": [0, 0, 0], "incorrect": [0, 0, 0]},
            "total_time_ms": 0.0,
        }
        self.filtering_time = 0.0
        self.ar_time = 0.0

        # 1. Load Configuration
        self._load_bci_config(config_path)

        # 2. Initialize Components
        self._init_models()
        self._init_buffers()
        self._init_sockets()

    def _load_bci_config(self, config_path: Path) -> None:
        """Loads and applies YAML configuration."""
        try:
            logger.info(f"Loading configuration: {config_path}")
            self.config = load_config(config_path)
            if hasattr(self.config, "random_state"):
                np.random.seed(self.config.random_state)
        except Exception as e:
            logger.error(f"Configuration Load Error: {e}")
            sys.exit(1)

    def _init_models(self) -> None:
        """Loads ML models, signal filters, and artifact rejection logic."""
        base_path = Path.cwd() / "resources" / "models"

        self.signal_filter = Filter(self.config, online=True)
        self.clf = RiemannianClf(cov_est="lwf").load(base_path / "model.pkl")
        self.controller = BCIController(self.config)

        with open(base_path / "ar.pkl", "rb") as f:
            self.ar = pickle.load(f)

    def _init_buffers(self) -> None:
        """Allocates data buffers based on window size and channel count."""
        self.keep_mask = [
            self.config.channels.index(ch)
            for ch in self.config.channels
            if self.config.remove_channels is not None
            and ch not in self.config.remove_channels
        ]

        n_channels = len(self.keep_mask)
        window_size = int(self.config.window_size)

        self.buffer = np.zeros((n_channels, window_size), dtype=np.float32)
        logger.info(
            f"Pipeline Buffer initialized: {n_channels}ch x {window_size} samples"
        )

    def _init_sockets(self) -> None:
        """Initializes ZMQ for visualization and UDP for external control."""
        self.zmq_ctx = zmq.Context()
        self.zmq_pub = self.zmq_ctx.socket(zmq.PUB)
        self.zmq_pub.bind(f"tcp://*:{ZMQ_PORT}")

        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info(f"Networking ready: ZMQ (PUB) port {ZMQ_PORT}")

    def _connect_streams(self) -> None:
        """Resolves LSL streams for EEG and Markers."""
        logger.info("Connecting to LSL streams...")
        streams = resolve_streams(wait_time=5.0)

        # Resolve EEG
        eeg_streams = [s for s in streams if s.type() == "EEG"]
        if not eeg_streams:
            raise RuntimeError("Required EEG stream not found.")
        self.inlet_eeg = StreamInlet(eeg_streams[0])

        # Resolve Markers (Optional ground truth)
        marker_name = (
            "MyDinoGameMarkerStream"
            if self.config.online == "dino"
            else "Labels_Stream"
        )
        label_streams = [
            s for s in streams if s.type() == "Markers" and s.name() == marker_name
        ]

        self.inlet_labels = StreamInlet(label_streams[0]) if label_streams else None
        if not self.inlet_labels:
            logger.warning(
                "No Marker Stream found; stats will be based on predictions only."
            )

    def _update_buffer(self, new_data: np.ndarray) -> None:
        """Shifts existing data in the buffer and appends new samples."""
        n_new = new_data.shape[1]
        if n_new == 0:
            return

        if n_new >= self.buffer.shape[1]:
            self.buffer = new_data[:, -self.buffer.shape[1] :]
        else:
            self.buffer[:, :-n_new] = self.buffer[:, n_new:]
            self.buffer[:, -n_new:] = new_data

    def run(self) -> None:
        """Main execution loop."""
        try:
            self._connect_streams()
        except RuntimeError as e:
            logger.error(e)
            return

        logger.info("BCI Engine Online.")
        current_label = UNKNOWN

        try:
            while self.running:
                loop_start = time.perf_counter()

                # 1. Pull Data
                chunk, _ = self.inlet_eeg.pull_chunk()
                if not chunk:
                    continue

                # 2. Update Ground Truth (if available)
                if self.inlet_labels:
                    marker_chunk, _ = self.inlet_labels.pull_chunk()
                    if marker_chunk:
                        current_label = marker_chunk[0][0]
                        if current_label not in CMD_MAP:
                            current_label = UNKNOWN

                # 3. Process Chunk
                # Scale: Dino mode expects Volts to Microvolts conversion
                eeg_data = np.array(chunk).T[self.keep_mask]
                if self.config.online == "dino":
                    eeg_data *= 1e-6

                f_start = time.perf_counter()
                filtered_chunk = self.signal_filter.apply_filter_online(eeg_data)
                self.filtering_time += (time.perf_counter() - f_start) * 1000

                self._update_buffer(filtered_chunk)
                self.zmq_pub.send_pyobj(("DATA", eeg_data, filtered_chunk))

                # 4. Artifact Rejection
                ar_start = time.perf_counter()
                is_artifact = self.ar.reject_bad_epochs_online(self.buffer)
                self.ar_time += (time.perf_counter() - ar_start) * 1000
                self.nr_of_loops += 1

                if is_artifact:
                    self.zmq_pub.send_pyobj(("EVENT", "Artifact"))
                    continue

                # 5. Classification & Robust Logic
                # Riemannian Predictor returns (label, probabilities, covariance)
                # COMMENT IF NORMAL CLASSIFIER IS USED
                prediction, probs, _ = self.clf.predict_with_recentering(self.buffer)

                # UNCOMMENT IF NORMAL CLASSIFIER IS USED
                # prediction, _ = self.clf.predict(self.buffer)
                # probs, _ = self.clf.predict_proba(self.buffer)

                if prediction is None or probs is None:
                    continue

                # Controller applies temporal smoothing and sends UDP commands
                smoothed_probs = self.controller.send_command(probs, self.udp_sock)

                if smoothed_probs is not None:
                    self.zmq_pub.send_pyobj(("PROBA", smoothed_probs.flatten()))
                    self._evaluate_performance(
                        prediction[0], smoothed_probs, current_label
                    )

                # 6. Global Timings
                self.stats["total_time_ms"] += (time.perf_counter() - loop_start) * 1000

        except KeyboardInterrupt:
            self.stop()

    def _evaluate_performance(
        self, pred: int, smoothed_probs: np.ndarray, label: str
    ) -> None:
        """Updates internal stats for post-run reporting."""
        if label == UNKNOWN:
            return

        ground_truth = CMD_MAP[label]
        self.stats["predictions"][ground_truth] += 1

        is_correct = pred == ground_truth
        is_accepted = np.max(smoothed_probs) >= self.config.classification_threshold

        category = "accepted" if is_accepted else "rejected"
        sub_cat = "correct" if is_correct else "incorrect"
        self.stats[category][sub_cat][ground_truth] += 1

    def _print_class_stats(self) -> None:
        """Prints a formatted ASCII table of the session performance."""
        class_names = ["Rest", "Left", "Right"]
        print("\n" + "=" * 85)
        print(f"{'ONLINE BCI PERFORMANCE REPORT':^85}")
        print("=" * 85)
        header = f"| {'Class':<12} | {'Total':<6} | {'Accepted (C/I)':<20} | {'Rejected':<10} | {'Rej Rate':<10} | {'Accuracy':<10} |"
        print(header + "\n" + "-" * 85)

        total_pred, total_corr_acc, total_acc = 0, 0, 0

        for i, name in enumerate(class_names):
            n_total = self.stats["predictions"][i]
            acc_c, acc_i = (
                self.stats["accepted"]["correct"][i],
                self.stats["accepted"]["incorrect"][i],
            )
            rej_c, rej_i = (
                self.stats["rejected"]["correct"][i],
                self.stats["rejected"]["incorrect"][i],
            )

            n_acc = acc_c + acc_i
            n_rej = rej_c + rej_i

            total_pred += n_total
            total_corr_acc += acc_c
            total_acc += n_acc

            acc_rate = (acc_c / n_acc * 100) if n_acc > 0 else 0.0
            rej_rate = (n_rej / n_total * 100) if n_total > 0 else 0.0

            print(
                f"| {name:<12} | {n_total:<6} | {f'{acc_c}/{acc_i}':<20} | {n_rej:<10} | {rej_rate:<9.1f}% | {acc_rate:<9.1f}% |"
            )

        print("-" * 85)
        global_acc = (total_corr_acc / total_acc * 100) if total_acc > 0 else 0.0
        print(
            f"| {'OVERALL':<12} | {total_pred:<6} | {'':<20} | {total_pred - total_acc:<10} | {((total_pred-total_acc)/total_pred*100):<9.1f}% | {global_acc:<9.1f}% |"
        )
        print("=" * 85 + "\n")

    def stop(self) -> None:
        """Gracefully shuts down sockets and prints diagnostic report."""
        self.running = False
        logger.info("Shutting down BCI Engine...")

        self._print_class_stats()

        if self.nr_of_loops > 0:
            avg_loop = self.stats["total_time_ms"] / max(
                1, sum(self.stats["predictions"])
            )
            print(f"Avg Processing Time per Epoch: {avg_loop:.2f} ms")
            print(
                f"Avg Filtering Time: {self.filtering_time / self.nr_of_loops:.2f} ms"
            )
            print(f"Avg Artifact Check: {self.ar_time / self.nr_of_loops:.2f} ms")

        self.udp_sock.close()
        self.zmq_ctx.destroy()


if __name__ == "__main__":
    conf_path = Path.cwd() / "resources" / "configs" / "bci_config.yaml"
    engine = BCIEngine(conf_path)
    engine.run()

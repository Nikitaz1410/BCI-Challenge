import json
import logging
import socket
import numpy as np
from typing import Optional, Tuple
from bci.utils.bci_config import EEGConfig

logger = logging.getLogger(__name__)


class BCIController:
    """
    Handles BCI decision-making logic and external communication.

    This controller smooths raw classifier probabilities using a sliding window
    buffer (Moving Average) and sends UDP commands to external applications
    (like a Dino Game) when confidence thresholds are met.
    """

    # Maps classifier index to (Human-readable Label, LSL/Event Marker)
    COMMAND_MAP = {
        0: ("CIRCLE", "CIRCLE ONSET"),
        1: ("ARROW LEFT", "ARROW LEFT ONSET"),
        2: ("ARROW RIGHT", "ARROW RIGHT ONSET"),
    }

    def __init__(self, config: EEGConfig) -> None:
        """
        Initializes the controller with a smoothing buffer.

        Args:
            config: Configuration object containing threshold, buffer size,
                    and network settings.
        """
        self.config = config
        self.threshold = self.config.classification_threshold

        # Initialize buffer with NaN to ensure we don't act until the window is full
        # Shape: (Number of Classes, Buffer Length)
        self.buffer_size = self.config.classification_buffer
        self.classification_buffer = np.full(
            (len(self.COMMAND_MAP), self.buffer_size), fill_value=np.nan
        )

    def send_command(
        self, probabilities: np.ndarray, sock: socket.socket
    ) -> Optional[np.ndarray]:
        """
        Processes new probabilities and sends a command if confidence is high enough.

        Args:
            probabilities: The latest probability vector from the classifier.
            sock: The UDP socket for communication.

        Returns:
            The integrated (averaged) probabilities if the buffer is full, else None.
        """
        # Ensure probabilities are aligned (vertical vector for the buffer)
        probs = probabilities.reshape(-1, 1)

        # 1. Update Sliding Window: Shift old values left and append new ones at the end
        self.classification_buffer = np.roll(self.classification_buffer, -1, axis=1)
        self.classification_buffer[:, -1:] = probs

        # 2. Guard: Only process if the buffer has been completely filled with data
        if np.isnan(self.classification_buffer).any():
            return None

        # 3. Integrate: Calculate the Moving Average across the time window
        avg_probs = np.mean(self.classification_buffer, axis=1)

        # 4. Decision: Find the class with the highest integrated probability
        best_class_idx = int(np.argmax(avg_probs))
        confidence = avg_probs[best_class_idx]

        # 5. Thresholding & Action
        if confidence >= self.threshold:
            self._execute_command(best_class_idx, confidence, sock)
        else:
            logger.debug(
                f"Confidence below threshold: {confidence:.2f} < {self.threshold}"
            )

        return avg_probs

    def _execute_command(
        self, class_idx: int, confidence: float, sock: socket.socket
    ) -> None:
        """Internal helper to format and send the UDP packet."""
        if class_idx not in self.COMMAND_MAP:
            logger.warning(f"Class index {class_idx} not found in COMMAND_MAP")
            return

        label, marker = self.COMMAND_MAP[class_idx]

        # Logic for the specific 'dino' game integration
        if self.config.online == "dino":
            payload = self._build_payload(label, marker)
            self._send_udp(sock, payload)

            logger.info(
                f"UDP SENT: {label} (Conf: {confidence:.2f}) -> "
                f"{self.config.ip}:{self.config.port}"
            )

    def _build_payload(self, label: str, marker: str) -> bytes:
        """Wraps the prediction into a JSON byte string."""
        payload = {"prediction": label, "marker_recent": marker}
        return json.dumps(payload).encode("utf-8")

    def _send_udp(self, sock: socket.socket, payload: bytes) -> None:
        """Dispatches the packet to the configured IP/Port."""
        try:
            sock.sendto(payload, (self.config.ip, self.config.port))
        except Exception as e:
            logger.error(f"Failed to send UDP packet: {e}")

    def reset_buffer(self) -> None:
        """Clears the smoothing buffer (useful between trials)."""
        self.classification_buffer.fill(np.nan)

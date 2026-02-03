import logging
import socket
import json
import numpy as np

from bci.utils.bci_config import EEGConfig

logger = logging.getLogger(__name__)


class BCIController:
    COMMAND_MAP = {
        0: ("CIRCLE", "CIRCLE ONSET"),
        1: ("ARROW LEFT", "ARROW LEFT ONSET"),
        2: ("ARROW RIGHT", "ARROW RIGHT ONSET"),
    }  # Map from classification outputs to commands

    def __init__(self, config: EEGConfig) -> None:
        self.config = config
        self.threshold = self.config.classification_threshold
        self.classification_buffer = np.full(
            (3, self.config.classification_buffer), fill_value=np.nan
        )

        # Connect to the marker stream coming from the dino game
        self._connect_to_marker_stream()

        # Only connect to a socket in case the dino game is expected
        # if self.config.online == "dino":
        # Connect to the dinogame UDP server to send commands
        # self._connect_socket(self.config.ip, self.config.port)

    def _connect_socket(self, ip: str, port: int):
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind((ip, port))
            server.listen()
            (client_socket, address) = server.accept()

            self.client_socket = client_socket
        except Exception:
            print("Error when connecting to the socket!")

    def _close_socket(self):
        try:
            self.client_socket.close()
        except Exception:
            print("Error when closing the socket!")

    def _connect_to_marker_stream(self):
        # MyDinoGameMarkerStream
        pass

    def _build_prediction(self, label: str, marker: str) -> bytes:
        payload = {"prediction": label, "marker_recent": marker}
        return json.dumps(payload).encode("utf-8")

    def _send_udp(
        self, sock: socket.socket, payload: bytes, ip: str, port: int
    ) -> None:
        sock.sendto(payload, (ip, port))

    def send_command(self, probabilities: np.ndarray, sock: socket.socket):
        probabilities = probabilities.T
        # Update the classification buffer
        self.classification_buffer[:, :-1] = self.classification_buffer[:, 1:]
        self.classification_buffer[:, -1:] = probabilities

        # Only process and send commands when buffer is full (no NaN values)
        if np.isnan(self.classification_buffer).any():
            return None

        # Integrate the probabilities
        # TODO: Check how to do this best. Right now it is an average
        buffer_probabilities = np.mean(self.classification_buffer, axis=1)
        # Select the most probable command (Highest integrated probability)
        most_probable_command = np.argmax(buffer_probabilities)

        # Check if the command confidence is over the manual threshold and only send the command if it is
        if buffer_probabilities[most_probable_command] >= self.threshold:
            label_marker = self.COMMAND_MAP[most_probable_command]
            if self.config.online == "dino":
                # Only send the command if the dino game is expected
                payload = self._build_prediction(*label_marker)
                self._send_udp(sock, payload, self.config.ip, self.config.port)
                logger.info(
                    f"UDP SENT: {label_marker[0]} (prob={buffer_probabilities[most_probable_command]:.2f}) "
                    f"-> {self.config.ip}:{self.config.port}"
                )
        else:
            logger.debug(
                f"Below threshold: max_prob={buffer_probabilities[most_probable_command]:.2f} < {self.threshold}"
            )

        return buffer_probabilities

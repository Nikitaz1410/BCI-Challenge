import socket
import json
import numpy as np

from bci.utils.bci_config import EEGConfig


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
        if np.isnan(self.classification_buffer).any():
            self.classification_buffer[:, :-1] = self.classification_buffer[:, 1:]
            self.classification_buffer[:, -1:] = probabilities
        else:
            self.classification_buffer[:, :-1] = self.classification_buffer[:, 1:]
            self.classification_buffer[:, -1:] = probabilities

            # Integrate the probabilities
            # TODO: Check how to do this best. Right now it is an average
            buffer_probabilities = np.mean(self.classification_buffer, axis=1)
            # Select the most probable command (Highest integrated probability)
            most_probable_command = np.argmax(buffer_probabilities)

            # Check if the command confidence is over the manual threshold and only send the command if it is
            # print(f"Buffer Probability: {buffer_probabilities}")
            # if buffer_probabilities[most_probable_command] >= self.threshold:

            label_marker = self.COMMAND_MAP[most_probable_command]
            if self.config.online == "dino":
                # Only send the command if the dino game is expected

                payload = self._build_prediction(*label_marker)
                # TODO: Check how to bind client to server
                self._send_udp(sock, payload, self.config.ip, self.config.port)

            return buffer_probabilities

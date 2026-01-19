import argparse
import json
import socket
import sys
import time

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows fallback
    msvcrt = None


UDP_IP = "127.0.0.1"
UDP_PORT = 5005

COMMAND_MAP = {
    "left": ("ARROW LEFT", "ARROW LEFT ONSET"),
    "right": ("ARROW RIGHT", "ARROW RIGHT ONSET"),
    "down": ("CIRCLE", "CIRCLE ONSET"),
}


def build_prediction(label: str, marker: str) -> bytes:
    payload = {"prediction": label, "marker_recent": marker}
    return json.dumps(payload).encode("utf-8")


def send_udp(sock: socket.socket, payload: bytes, ip: str, port: int) -> None:
    sock.sendto(payload, (ip, port))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive UDP sender for Dino quicktime trials.")
    parser.add_argument("--ip", default=UDP_IP, help="Destination IP (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=UDP_PORT, help="Destination UDP port (default: 5005).")
    return parser.parse_args(argv)


def _poll_key_windows() -> str | None:
    if not msvcrt or not msvcrt.kbhit():
        return None
    code = msvcrt.getwch()
    if code in ("q", "Q"):
        return "quit"
    if code in ("\x00", "\xe0"):
        arrow = ord(msvcrt.getwch())
        if arrow == 75:
            return "left"
        if arrow == 77:
            return "right"
        if arrow == 80:
            return "down"
    return None


def _poll_key_posix() -> str | None:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
    if not rlist:
        return None

    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = sys.stdin.read(1)
        if first in ("q", "Q"):
            return "quit"
        if first != "\x1b":
            return None
        rest = sys.stdin.read(2)
        if rest == "[D":
            return "left"
        if rest == "[C":
            return "right"
        if rest == "[B":
            return "down"
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def poll_key() -> str | None:
    if msvcrt is not None:
        return _poll_key_windows()
    return _poll_key_posix()


def interactive_loop(ip: str, port: int) -> None:
    print("Press LEFT, RIGHT, or DOWN to send a command. Press Q to quit.")
    print(f"Sending to {ip}:{port}.")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        while True:
            key = poll_key()
            if key is None:
                time.sleep(0.01)
                continue
            if key == "quit":
                print("Exiting sender.")
                return
            label_marker = COMMAND_MAP.get(key)
            if not label_marker:
                continue
            payload = build_prediction(*label_marker)
            send_udp(sock, payload, ip, port)
            print(f"Sent {label_marker[0]} -> {payload.decode('utf-8')}")


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    try:
        interactive_loop(args.ip, args.port)
    except KeyboardInterrupt:
        print("Interrupted; exiting.")


if __name__ == "__main__":
    main(sys.argv[1:])

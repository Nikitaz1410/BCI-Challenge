# helpers.py
# -*- coding: utf-8 -*-
"""
Utilities:
- YAML config loader
- UDP listener + blink/classifier helpers
- Embedded PyQt5 Survey + Reaction test
- Qt→Pygame bridge (renders Qt into the current Pygame display)
"""

from __future__ import annotations

# =============================================================================
# Standard Library
# =============================================================================
import json
import os
import random
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Any

# =============================================================================
# Third-party
# =============================================================================
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from pylsl import StreamInfo, StreamOutlet
import pygame  # only for display + event key mappings

# =============================================================================
# Config & Runtime Context
# =============================================================================


class config:
    """Load simple game settings from YAML (unchanged fields)."""

    def __init__(self) -> None:
        cwd_path = os.getcwd()
        config_path = os.path.join(
            cwd_path, "resources", "configs", "dino_config.yaml"
        )  # Our
        # config_path = os.path.join(cwd_path, "config", "dino_config.yaml") # Old
        with open(config_path, "r") as file:
            cfg = yaml.safe_load(file)
        self.countdown = cfg["Countdown"]  # seconds
        self.success_rate = cfg["success_rate"]  # difficulty
        self.game_speed = cfg["game_speed"]  # speed
        self.max_time = cfg["quicktime_duration"]  # seconds
        self.num_reaction_times = cfg["num_reaction_times"]
        self.break_questions = cfg["break_questions_de"]  # list of dicts
        self.questions = cfg["questions"]  # list of dicts
        self.num_tasks = cfg["num_tasks"]  # number of tasks
        self.day_time = cfg["day_time"]  # True/False
        self.sleep_time = cfg.get("sleep_time", 0.15)  # seconds to sleep between bumps
        self.debug_hud = cfg.get("debug_hud", False)  # enable debug HUD
        self.stale_pred_s = cfg.get("stale_pred_s", 1.0)  # seconds

        self.evidence_window_s = cfg.get("evidence_window_s", 0.5)  # seconds
        self.ignore_first_s = cfg.get("ignore_first_s", 0.2)  # seconds
        self.min_votes = cfg.get("min_votes", 2)
        self.refractory_s = cfg.get("refractory_s", 0.25)
        self.gated_bump_multiplier = cfg.get("gated_bump_multiplier", 2)
        self.prob_threshold = cfg.get("prob_threshold", 0.5)
        self.use_replay_markers = cfg.get("use_replay_markers", False)


@dataclass(frozen=True)
class MentalConfig:
    mapping_table: dict

    cue_list: list


# Runtime values set by main/menu
_SURVEY_LANGUAGE: str = "eng"
_USER_ID: str = "P004"
_CWD_PATH: str = os.getcwd()


def set_runtime_context(user_id: str, cwd_path: str, language: str) -> None:
    """Set globals used by the SurveyApp."""
    global _USER_ID, _CWD_PATH, _SURVEY_LANGUAGE
    _USER_ID = user_id
    _CWD_PATH = cwd_path
    _SURVEY_LANGUAGE = language


# =============================================================================
# UDP / Classifier / Blink
# =============================================================================

BLINK_CONF_THRESHOLD = 0.60
BLINK_REFRACTORY_MS = 250
CMD_RECENT_MS = 400  # ms

# entries: (mono_time, confidence, remote_ts)
blink_queue: Deque[Tuple[float, float, float]] = deque(maxlen=64)
_last_blink_mono: float = 0.0

recent_cmd_idx: Optional[int] = None  # 0: circle, 1: left, 2: right
recent_cmd_label: str = ""
recent_cmd_mono: float = 0.0  # perf_counter of last classifier command

LABEL2IDX: Dict[str, int] = {
    "CIRCLE": 0,
    "CIRCLE ONSET": 0,
    "circle": 0,
    "Circle": 0,
    "LEFT": 1,
    "ARROW LEFT": 1,
    "ARROW LEFT ONSET": 1,
    "left": 1,
    "Left": 1,
    "RIGHT": 2,
    "ARROW RIGHT": 2,
    "ARROW RIGHT ONSET": 2,
    "right": 2,
    "Right": 2,
}

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
udp_connected = False
udp_lock = threading.Lock()
recieved_something = False  # True if we received any UDP message
recent_marker: Optional[str] = None  # last marker received from UDP
recent_packets: Deque[Tuple[str, float, bool, str]] = deque(maxlen=5)
_udp_thread_obj: Optional[threading.Thread] = None
_udp_stop_event = threading.Event()
last_payload_text: str = ""
last_payload_time: float = 0.0
last_payload_recognized: bool = False
last_payload_label: str = ""
PAYLOAD_RETENTION_S = 2.0


def _map_prediction_to_idx(pred: Any) -> Optional[int]:
    """Robust mapping from numeric or string prediction to cue index."""
    try:
        n = int(pred)
        if n in (0, 1, 2):
            return n
    except Exception:
        pass
    if isinstance(pred, str):
        key = pred.strip()
        return LABEL2IDX.get(key, LABEL2IDX.get(key.upper()))
    return None


def poll_blink() -> Optional[Tuple[float, float, float]]:
    """Pop one blink from the queue, or None."""
    with udp_lock:
        if blink_queue:
            return blink_queue.popleft()
    return None


def get_recent_cmd_idx(max_age_ms: int = CMD_RECENT_MS) -> Optional[int]:
    """Return last classifier command index if recent enough, else None."""
    global recieved_something
    now = time.perf_counter()
    with udp_lock:
        if recent_cmd_idx is None:
            return None
        if (now - recent_cmd_mono) * 1000.0 <= max_age_ms:
            recieved_something = True
            return recent_cmd_idx
        return None


def get_recent_cmd_details(
    max_age_ms: int = CMD_RECENT_MS,
) -> tuple[Optional[int], Optional[str], Optional[float]]:
    """Return (idx, label, mono_time) for the most recent classifier command."""
    now = time.perf_counter()
    with udp_lock:
        if recent_cmd_idx is None:
            return None, None, None
        if (now - recent_cmd_mono) * 1000.0 <= max_age_ms:
            return recent_cmd_idx, recent_cmd_label, recent_cmd_mono
        return None, None, None


def get_current_marker() -> Optional[str]:
    """Return the most recent command label, or None if not set."""
    global recieved_something
    with udp_lock:
        if recent_marker:
            recieved_something = True
            return recent_marker
    return None


def get_prediction_udp_connected() -> bool:
    """Check if the UDP listener is connected."""
    return recieved_something


def get_udp_listener_status() -> (
    tuple[bool, bool, int, list[Tuple[str, float, bool, str]]]
):
    """Return (listener_bound, has_received_data, port, recent_packets).

    Each recent packet entry is (payload_text, timestamp, recognized, label).
    """
    with udp_lock:
        packets = list(recent_packets)
        has_data = recieved_something or bool(packets)
        thread_alive = _udp_thread_obj is not None and _udp_thread_obj.is_alive()
        return udp_connected and thread_alive, has_data, UDP_PORT, packets


def configure_udp_endpoint(port: int) -> bool:
    """Set the UDP port before the listener thread starts."""
    global UDP_PORT, udp_connected, recieved_something
    global \
        last_payload_text, \
        last_payload_time, \
        last_payload_label, \
        last_payload_recognized
    with udp_lock:
        if is_udp_thread_running():
            return False
        UDP_PORT = port
        udp_connected = False
        recieved_something = False
        last_payload_text = ""
        last_payload_time = 0.0
        recent_packets.clear()
        last_payload_label = ""
        last_payload_recognized = False
        return True


def is_udp_thread_running() -> bool:
    """Return True if the UDP listener thread is alive."""
    return _udp_thread_obj is not None and _udp_thread_obj.is_alive()


def _record_payload_locked(
    payload_text: str, recognized: bool = False, label: str = ""
) -> None:
    """Record the latest UDP payload under lock for debugging UI."""
    global \
        last_payload_text, \
        last_payload_time, \
        last_payload_recognized, \
        last_payload_label
    ts = time.time()
    recent_packets.append((payload_text, ts, recognized, label))
    last_payload_text = payload_text
    last_payload_time = ts
    last_payload_recognized = recognized
    last_payload_label = label


def get_udp_payload_snapshot() -> tuple[str, float, bool, str]:
    """Return details about the most recent UDP payload."""
    global \
        last_payload_text, \
        last_payload_time, \
        last_payload_recognized, \
        last_payload_label
    with udp_lock:
        now = time.time()
        if last_payload_time and (now - last_payload_time) > PAYLOAD_RETENTION_S:
            last_payload_text = ""
            last_payload_time = 0.0
            last_payload_recognized = False
            last_payload_label = ""
        return (
            last_payload_text,
            last_payload_time,
            last_payload_recognized,
            last_payload_label,
        )


def _udp_thread() -> None:
    """Background UDP listener for blinks and classifier events."""
    global udp_connected, _last_blink_mono, recieved_something
    global recent_cmd_idx, recent_cmd_label, recent_cmd_mono, recent_marker

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
    try:
        sock.bind((UDP_IP, UDP_PORT))
        udp_connected = True
        print(f"UDP listener started on {UDP_IP}:{UDP_PORT}")
        while not _udp_stop_event.is_set():
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            msg = data.decode("utf-8", errors="ignore").strip()
            # JSON or legacy plain text
            try:
                payload = json.loads(msg)
                # print(f"Received JSON payload: {payload}")
            except Exception:
                payload = None

            if payload is None:
                idx = _map_prediction_to_idx(msg)
                msg_text = str(msg)
                recognized = idx is not None
                with udp_lock:
                    recent_cmd_idx = idx
                    recent_cmd_label = msg_text
                    recent_cmd_mono = time.perf_counter()
                    if recognized:
                        recieved_something = True
                    _record_payload_locked(msg_text, recognized, msg_text)
                continue

            if isinstance(payload, (int, float, str)):
                idx = _map_prediction_to_idx(payload)
                payload_text = str(payload)
                recognized = idx is not None
                with udp_lock:
                    recent_cmd_idx = idx
                    recent_cmd_label = payload_text
                    recent_cmd_mono = time.perf_counter()
                    if recognized:
                        recieved_something = True
                    _record_payload_locked(payload_text, recognized, payload_text)
                continue

            if not isinstance(payload, dict):
                # Unhandled payload type; skip
                continue

            evt = str(payload.get("event", "")).lower()

            if evt == "blink":
                conf = float(payload.get("confidence", 0.0))
                rts = float(payload.get("timestamp", 0.0))
                now_mono = time.perf_counter()
                if (now_mono - _last_blink_mono) * 1000.0 < BLINK_REFRACTORY_MS:
                    continue
                _last_blink_mono = now_mono
                with udp_lock:
                    blink_queue.append((now_mono, conf, rts))
                    recieved_something = True
                    _record_payload_locked(msg, False, "blink")
                print("BLINK → queued in blink_queue:", conf, rts)  # DEBUG
                continue

            # classifier (or fallback if event missing)
            pred = payload.get("prediction", payload.get("label"))
            idx = _map_prediction_to_idx(pred)
            label_text = str(payload.get("label", pred))
            recognized = idx is not None
            with udp_lock:
                recent_cmd_idx = idx
                recent_cmd_label = label_text
                recent_marker = str(payload.get("marker_recent", ""))
                # print(f"Received classifier command: {recent_cmd_label} (idx: {recent_cmd_idx})")
                recent_cmd_mono = time.perf_counter()
                if recognized:
                    recieved_something = True
                _record_payload_locked(msg, recognized, label_text)
    except Exception as e:
        print(f"UDP listener error: {e}")
        udp_connected = False
    finally:
        try:
            sock.close()
        except Exception:
            pass
        udp_connected = False


def start_udp_thread() -> None:
    global _udp_thread_obj
    if is_udp_thread_running():
        return
    _udp_stop_event.clear()
    thread = threading.Thread(target=_udp_thread, daemon=True)
    _udp_thread_obj = thread
    thread.start()


def stop_udp_thread(timeout: float = 2.0) -> bool:
    """Request the UDP listener to stop and wait for completion."""
    global _udp_thread_obj, udp_connected
    if not is_udp_thread_running():
        return True
    _udp_stop_event.set()
    thread = _udp_thread_obj
    if thread is not None:
        thread.join(timeout=timeout)
    with udp_lock:
        udp_connected = False
    still_running = is_udp_thread_running()
    if not still_running:
        _udp_stop_event.clear()
        _udp_thread_obj = None
    return not still_running


# =============================================================================
# PyQt widgets
# =============================================================================


class SurveyApp(QtWidgets.QMainWindow):
    """GUI for survey + reaction test (unchanged logic)."""

    finished = QtCore.pyqtSignal()  # emitted after reaction time saved

    def __init__(self, cwd_path: str, part_number: int, cur_session: int) -> None:
        super().__init__()
        start_udp_thread()
        self.setWindowTitle("Survey Application")

        # --- runtime ---
        self.cwdPath = cwd_path
        self.part_number = part_number
        self.cur_session = cur_session

        # --- LSL stream ---
        self.lsl_info = StreamInfo(
            "Metadata", "meta", 1, 0, "string", "DinoParadigm12345"
        )
        self.lsl_outlet = StreamOutlet(self.lsl_info)

        # --- state ---
        self.reactiongame_started = False
        self.reaction_start_time: Optional[float] = None
        self.cnt_num_reactiontest = 1

        self.responses: Dict[int, str] = {}
        self.current_page = 0
        self.current_run = 1
        self.current_trial = 0
        self.stimuli_sequence: List[Any] = []
        self.in_arrow_paradigm = False
        self.break_questions_start = True
        self.questions_len = 0
        self.current_stimulus: Optional[Any] = None

        # --- config ---
        self.num_repetitions = config().num_reaction_times
        self.break_questions = config().break_questions
        self.questions = config().questions

        # --- UI ---
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet(
            "background-color: #000000; color: white; font-family: Arial;"
        )
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.layout.setSpacing(40)

        self.question_label = QtWidgets.QLabel(
            alignment=QtCore.Qt.AlignCenter, wordWrap=True
        )
        self.question_label.setStyleSheet("font-size: 36px; margin-bottom: 20px;")
        self.layout.addWidget(self.question_label)

        self.image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.input_widget: Optional[QtWidgets.QWidget] = None
        self.slider_label: Optional[QtWidgets.QLabel] = None
        self.btn_group: Optional[QtWidgets.QButtonGroup] = None

        self.nav_layout = QtWidgets.QHBoxLayout()
        self.nav_layout.setSpacing(60)

        self.prev_button = self._make_button("Previous", self.previous_page)
        self.next_button = self._make_button("Next", self.next_page)

        self.submit_button = self._make_button(
            "Submit", self.submit, css="background-color: green; color: white;"
        )
        self.submit_button.hide()

        self.layout.addLayout(self.nav_layout)

        # Reaction test controls
        start_text = (
            "Start Reaction Test"
            if _SURVEY_LANGUAGE == "eng"
            else "Reaktionstest starten"
        )
        self.start_reactiontest_button = self._make_wide_button(
            start_text, self.start_reaction_time, blue=True
        )
        self.start_reactiontest_button.hide()
        self.layout.addWidget(
            self.start_reactiontest_button, alignment=QtCore.Qt.AlignCenter
        )

        self.click_button = self._make_wide_button(
            "Click!", self._reaction_blink, tall=True, dark=True
        )
        self.click_button.hide()
        self.layout.addWidget(self.click_button, alignment=QtCore.Qt.AlignCenter)

        save_text = (
            "Save reaction time"
            if _SURVEY_LANGUAGE == "eng"
            else "Reaktionszeit speichern"
        )
        self.accept_reaction_button = self._make_wide_button(
            save_text, self.save_reaction_time, green=True
        )
        self.accept_reaction_button.hide()
        self.layout.addWidget(
            self.accept_reaction_button, alignment=QtCore.Qt.AlignCenter
        )

        # Blink polling
        self._blink_poll_timer = QTimer(self)
        self._blink_poll_timer.timeout.connect(self._poll_blinks)
        self._blink_poll_timer.start(10)  # ~100 Hz
        self.awaiting_start_blink = False
        self.reaction_started = False
        # NEW: dedicated timer for switching to GREEN (so we can cancel it)
        self._green_timer = QTimer(self)
        self._green_timer.setSingleShot(True)
        self._green_timer.timeout.connect(self.go_green)

        # NEW: Retry button (do not save, try again)
        retry_text = (
            "Retry (don’t save)"
            if _SURVEY_LANGUAGE == "eng"
            else "Erneut (nicht speichern)"
        )
        self.retry_button = self._make_wide_button(
            retry_text, self.retry_reaction_time, dark=True
        )
        self.retry_button.hide()
        self.layout.addWidget(self.retry_button, alignment=QtCore.Qt.AlignCenter)

        # ---- double-blink-to-save state ----
        self._awaiting_save_by_dblblink = False  # if True, look for two blinks to save
        self._dbl_first_ts: Optional[float] = None
        self._dbl_gap_ms = (
            600  # max gap between two blinks to count as "double-blink save"
        )
        self.load_page()

    # ----- small UI helpers -------------------------------------------------
    def _make_button(
        self, text: str, handler, css: str = "background-color: darkgray;"
    ) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setStyleSheet(
            f"font-size: 28px; height: 100px; width: 300px; border-radius: 15px; {css}"
        )
        btn.clicked.connect(handler)
        self.nav_layout.addWidget(btn)
        return btn

    def _make_wide_button(
        self, text: str, handler, *, blue=False, green=False, dark=False, tall=False
    ) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        bg = (
            "blue"
            if blue
            else ("green" if green else ("black" if dark else "darkgray"))
        )
        height = "220px" if tall else "120px"
        btn.setStyleSheet(
            f"font-size: 36px; height: {height}; width: 600px; background-color: {bg}; color: white; border-radius: 20px;"
        )
        btn.clicked.connect(handler)
        return btn

    # ----- survey flow ------------------------------------------------------
    def reset_widgets(self) -> None:
        self.question_label.show()
        self.image_label.hide()
        self.start_reactiontest_button.hide()
        if self.input_widget:
            self.input_widget.hide()
        if self.slider_label:
            self.slider_label.hide()
        self.submit_button.show()
        self.prev_button.show()
        self.next_button.show()

    def load_page(self) -> None:
        self.reset_widgets()

        current = None
        if self.break_questions_start:
            current = self.break_questions[self.current_page]
            self.questions_len = len(self.break_questions)

        if self.current_page < self.questions_len:
            self.question_label.setText(current["question"])
            qtype = current["type"]

            if self.input_widget:
                self.layout.removeWidget(self.input_widget)
                self.input_widget.deleteLater()
                self.input_widget = None

            if qtype == "entry":
                self._setup_entry(current)
            elif qtype == "options":
                self._setup_options(current)
            elif qtype in ("range", "range-text"):
                self._setup_range(current, text_mode=(qtype == "range-text"))

        self._maybe_show_image(current)
        self._update_nav_visibility()

    def _setup_entry(self, q: Dict[str, Any]) -> None:
        w = QtWidgets.QLineEdit()
        w.setText(q.get("default", ""))
        w.setStyleSheet(
            "font-size: 28px; height: 60px; width: 600px; background-color: grey; color: black; border-radius: 15px; padding: 10px;"
        )
        w.setAlignment(QtCore.Qt.AlignCenter)
        self.input_widget = w
        self.layout.insertWidget(2, self.input_widget, alignment=QtCore.Qt.AlignCenter)

    def _setup_options(self, q: Dict[str, Any]) -> None:
        gb = QtWidgets.QGroupBox()
        gb.setAlignment(QtCore.Qt.AlignCenter)
        gb.setStyleSheet(
            "QGroupBox {background:#111; color:white; border-radius:10px; padding:15px;}"
            "QRadioButton {font-size:24px; height:50px; width:500px; background:grey; color:black; border-radius:10px; padding:5px;}"
            "QRadioButton::indicator:checked {background:#ffcc00;}"
        )
        vbox = QtWidgets.QVBoxLayout(gb)
        vbox.setSpacing(20)
        vbox.setAlignment(QtCore.Qt.AlignCenter)
        self.btn_group = QtWidgets.QButtonGroup(self)
        self.btn_group.setExclusive(True)
        for index, answer in enumerate(q.get("set_answers", []), start=1):
            rb = QtWidgets.QRadioButton(answer)
            self.btn_group.addButton(rb, index)
            vbox.addWidget(rb)
        if self.btn_group.buttons():
            self.btn_group.buttons()[0].setChecked(True)
        self.btn_group.idToggled.connect(self.on_option_chosen)
        self.input_widget = gb
        self.layout.insertWidget(2, gb, alignment=QtCore.Qt.AlignCenter)

    def _setup_range(self, q: Dict[str, Any], *, text_mode: bool) -> None:
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        min_value, max_value = q.get("min", 1), q.get("max", 5)
        slider.setRange(min_value, max_value)
        slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.setStyleSheet(
            "height: 60px; width: 900px; background-color: #111; color: white; border-radius: 5px; padding: 6px;"
        )
        slider.setValue((min_value + max_value) // 2)
        self.input_widget = slider

        if text_mode:
            self.answer_set = q.get("set_answers", [])
            txt = (
                self.answer_set[slider.value() - 1]
                if self.answer_set
                else str(slider.value())
            )
            slider.valueChanged.connect(self.update_slider_label_text)
            self.slider_label = QtWidgets.QLabel(f"Value: {txt}")
        else:
            slider.valueChanged.connect(self.update_slider_label)
            self.slider_label = QtWidgets.QLabel(f"Value: {slider.value()}")

        self.slider_label.setAlignment(QtCore.Qt.AlignCenter)
        self.slider_label.setStyleSheet(
            "font-size: 24px; color: #ffcc00; margin-top: 10px;"
        )
        self.layout.insertWidget(2, self.slider_label, alignment=QtCore.Qt.AlignCenter)
        self.layout.insertWidget(2, slider, alignment=QtCore.Qt.AlignCenter)

    def _maybe_show_image(self, q: Optional[Dict[str, Any]]) -> None:
        if q and q.get("image"):
            path = os.path.join(self.cwdPath, "assets", q["image"])
            if os.path.exists(path):
                pixmap = QtGui.QPixmap(path).scaled(
                    1200, 900, QtCore.Qt.KeepAspectRatio
                )
                self.image_label.setPixmap(pixmap)
                self.image_label.show()
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Image file not found: {path}"
                )
                self.image_label.hide()
        else:
            self.image_label.hide()

    def _update_nav_visibility(self) -> None:
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setVisible(self.current_page < self.questions_len - 1)
        self.submit_button.setVisible(self.current_page == self.questions_len - 1)

    def update_slider_label(self) -> None:
        if self.slider_label and isinstance(self.input_widget, QtWidgets.QSlider):
            self.slider_label.setText(f"Value: {self.input_widget.value()}")

    def update_slider_label_text(self) -> None:
        if self.slider_label and isinstance(self.input_widget, QtWidgets.QSlider):
            idx = self.input_widget.value() - 1
            txt = (
                self.answer_set[idx]
                if 0 <= idx < len(getattr(self, "answer_set", []))
                else str(self.input_widget.value())
            )
            self.slider_label.setText(f"Value: {txt}")

    def on_option_chosen(self, btn_id: int, checked: bool) -> None:
        if checked and isinstance(self.input_widget, QtWidgets.QGroupBox):
            chosen_text = self.btn_group.button(btn_id).text()
            self.input_widget.setTitle(f"Selected: {chosen_text}")

    def save_response(self) -> bool:
        """Collect response from current input widget."""
        if self.current_page < self.questions_len:
            if isinstance(self.input_widget, QtWidgets.QLineEdit):
                response = self.input_widget.text()
            elif isinstance(self.input_widget, QtWidgets.QComboBox):
                response = self.input_widget.currentText()
            elif isinstance(self.input_widget, QtWidgets.QSlider):
                response = str(self.input_widget.value())
            elif isinstance(self.input_widget, QtWidgets.QGroupBox):
                sel = self.btn_group.checkedButton() if self.btn_group else None
                response = sel.text() if sel else ""
            else:
                response = ""

            if not response.strip():
                msg = (
                    "Please provide an answer before proceeding."
                    if _SURVEY_LANGUAGE == "eng"
                    else "Bitte geben Sie eine Antwort ein, bevor Sie fortfahren."
                )
                QtWidgets.QMessageBox.warning(self, "Error", msg)
                return False

            self.responses[self.current_page] = response
            return True
        return True

    def next_page(self) -> None:
        if self.save_response():
            self.current_page += 1
            self.load_page()

    def previous_page(self) -> None:
        self.current_page = max(0, self.current_page - 1)
        self.load_page()

    def submit(self) -> None:
        """Save survey CSV, then reveal reaction test start button."""
        if not self.save_response():
            return

        # LSL marker (survey end)
        try:
            marker = self.config.get("Markers", {}).get("survey_end", "SURVEY_END")
            self.lsl_outlet.push_sample([marker])
        except Exception:
            pass

        # CSV save
        try:
            import csv

            root_project = _CWD_PATH
            file_path = os.path.join(
                root_project, "data", f"sub-{_USER_ID}", "survey_dino_new.csv"
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file_exists = os.path.isfile(file_path)
            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if self.break_questions_start:
                    if not file_exists:
                        writer.writerow([q["question"] for q in self.break_questions])
                    writer.writerow(
                        [
                            self.responses.get(i, "")
                            for i in range(len(self.break_questions))
                        ]
                    )
                else:
                    if not file_exists:
                        writer.writerow([q["question"] for q in self.questions])
                    writer.writerow(
                        [self.responses.get(i, "") for i in range(len(self.questions))]
                    )

            # switch to reaction test intro
            self.question_label.hide()
            self.image_label.hide()
            if self.input_widget:
                self.input_widget.hide()
            if self.slider_label:
                self.slider_label.hide()
            self.submit_button.hide()
            self.prev_button.hide()
            self.next_button.hide()
            # self.start_reactiontest_button.show()
            self.break_questions_start = True

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to save the survey results: {e}"
            )

    # ----- Reaction test (blink-driven) -------------------------------------
    def start_reaction_time(self) -> None:
        self.retry_button.hide()
        self.start_reactiontest_button.hide()
        self.accept_reaction_button.hide()
        self.image_label.clear()

        self.reactiongame_started = False
        self.reaction_started = False
        self.setWindowTitle("Reaction Time Test")

        self.central_widget.setStyleSheet(
            "background-color: black; color: white; font-family: Arial;"
        )
        self.image_label.setStyleSheet(
            "font-size: 48px; font-weight: bold; color: white;"
        )
        self.image_label.setText(
            "Blink to start" if _SURVEY_LANGUAGE == "eng" else "Blinzeln, um zu starten"
        )
        self.image_label.show()
        self.click_button.hide()

        self.awaiting_start_blink = True

    def go_green(self) -> None:
        self.reaction_started = True
        self.central_widget.setStyleSheet(
            "background-color: green; color: white; font-family: Arial;"
        )
        self.image_label.setText(
            "BLINK now!" if _SURVEY_LANGUAGE == "eng" else "JETZT blinzeln!"
        )
        self.start_time = time.perf_counter()

    def save_reaction_time(self) -> None:
        # Disarm double-blink state (if still armed)
        self._awaiting_save_by_dblblink = False
        self._dbl_first_ts = None

        self.accept_reaction_button.hide()
        self.retry_button.hide()  # NEW
        self.image_label.clear()

        msg = (
            "Reaction time saved!\n Saved "
            if _SURVEY_LANGUAGE == "eng"
            else "Reaktionszeit gespeichert!\n Gespeichert "
        )
        self.image_label.setText(
            f"{msg}{self.cnt_num_reactiontest}/{self.num_repetitions}"
        )
        self.cnt_num_reactiontest += 1
        self.image_label.setStyleSheet(
            "font-size: 48px; font-weight: bold; color: white;"
        )
        self.image_label.show()

        # save CSV
        file_path = os.path.join(
            os.getcwd(), "data", f"sub-{_USER_ID}", "reaction_times.csv"
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode="a", newline="") as file:
            import csv, time as _t

            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "trial", "reaction_time"])
            writer.writerow([_t.time(), self.cnt_num_reactiontest, self.reaction_time])

        # send lsl marker with reaction time
        self.lsl_outlet.push_sample(
            [f"reaction_{self.cnt_num_reactiontest},{self.reaction_time}"]
        )

        # completion check
        if self.cnt_num_reactiontest >= self.num_repetitions:
            end_msg = (
                "Reaction test complete! \n Dino game starts in 5 seconds..."
                if _SURVEY_LANGUAGE == "eng"
                else "Reaktionstest abgeschlossen! \n Dino-Spiel beginnt in 5 Sekunden..."
            )
            self.image_label.setText(end_msg)
            self.image_label.show()
            self.awaiting_start_blink = True
            self.reactiongame_started = False
            self.reaction_started = False
            self.cnt_num_reactiontest = 1

            self.start_reactiontest_button.hide()
            try:
                self.lsl_outlet.push_sample(
                    [self.config["Markers"]["reactiontest_end"]]
                )
            except Exception:
                pass
            self.finished.emit()  # signal completion
        else:
            QTimer.singleShot(2000, self._start_trial_after_blink)

    def _start_trial_after_blink(self) -> None:
        """Begin one trial (RED → random delay → GREEN)."""
        self.accept_reaction_button.hide()
        self.retry_button.hide()

        self.reactiongame_started = True
        self.reaction_started = False
        self.central_widget.setStyleSheet(
            "background-color: red; color: white; font-family: Arial;"
        )
        self.image_label.setStyleSheet(
            "font-size: 48px; font-weight: bold; color: white;"
        )
        self.image_label.setText(
            "Wait until green! (blink only when green)"
            if _SURVEY_LANGUAGE == "eng"
            else "Warte bis grün! (nur bei grün blinzeln)"
        )

        # REPLACE singleShot with controlled member timer
        self._green_timer.stop()
        self._green_timer.start(random.randint(1500, 4000))
        try:
            self.lsl_outlet.push_sample(
                [self.config["Markers"].get("reactiontest_start", "reactiontest_start")]
            )
        except Exception:
            pass

    def _handle_false_start(self) -> None:
        """Reset cleanly after a false start (blink or button)."""
        # stop any pending GREEN transition from the previous trial
        try:
            self._green_timer.stop()
        except Exception:
            pass

        # fully reset state
        self.reactiongame_started = False
        self.reaction_started = False

        # brief feedback
        self.central_widget.setStyleSheet(
            "background-color: grey; color: white; font-family: Arial;"
        )
        txt = (
            "Too early! Blink only when green."
            if _SURVEY_LANGUAGE == "eng"
            else "Zu früh! Nur bei grün blinzeln."
        )
        self.image_label.setText(txt)

        # IMPORTANT: drop stale blink events so we don't instantly re-trigger
        try:
            with udp_lock:
                blink_queue.clear()
        except Exception:
            pass

        # after a short pause, go back to the *arm* screen: "Blink to start"
        def _rearm_to_blink_start():
            self.central_widget.setStyleSheet(
                "background-color: black; color: white; font-family: Arial;"
            )
            self.image_label.setText(
                "Blink to start"
                if _SURVEY_LANGUAGE == "eng"
                else "Blinzeln, um zu starten"
            )
            self.awaiting_start_blink = True
            self.click_button.hide()  # hide button in blink mode

        QTimer.singleShot(800, _rearm_to_blink_start)

    def retry_reaction_time(self) -> None:
        """
        Discard the just-measured reaction time (do not save or increment trial),
        and re-arm the test according to the active mode. Also cancel any pending
        GREEN transition to avoid accidental continuation.
        """
        # Stop any scheduled GREEN switch from a previous arm
        # Also disarm any double-blink save state
        self._awaiting_save_by_dblblink = False
        self._dbl_first_ts = None

        try:
            self._green_timer.stop()
        except Exception:
            pass

        # Reset internal state
        self.reactiongame_started = False
        self.reaction_started = False

        # Hide decision buttons
        self.accept_reaction_button.hide()
        self.retry_button.hide()

        # (Optional) Send an LSL marker to note the discard
        try:
            self.lsl_outlet.push_sample(["reaction_discard"])
        except Exception:
            pass

        # Clear any queued blinks so we don't immediately re-trigger
        try:
            with udp_lock:
                blink_queue.clear()
        except Exception:
            pass

        # Re-arm per mode
        self.central_widget.setStyleSheet(
            "background-color: black; color: white; font-family: Arial;"
        )

        # Back to “Blink to start”
        self.image_label.setText(
            "Blink to start" if _SURVEY_LANGUAGE == "eng" else "Blinzeln, um zu starten"
        )
        self.click_button.hide()
        self.awaiting_start_blink = True

    def _reaction_blink(self, remote_ts: float) -> None:
        """Blink received while GREEN: compute RT and show save button."""
        self.reactiongame_started = False
        if not self.reaction_started:
            self._handle_false_start()
            return
        self.reaction_started = False

        self.reaction_time = int((remote_ts - self.start_time) * 1000)
        self.central_widget.setStyleSheet(
            "background-color: black; color: white; font-family: Arial;"
        )

        # Show RT and instruct "double-blink to save"; keep Retry button visible
        msg_rt = (
            "Your reaction time: "
            if _SURVEY_LANGUAGE == "eng"
            else "Deine Reaktionszeit: "
        )
        msg_save = (
            "\n\nDouble-blink to save (≤ 600 ms), \nor press Retry."
            if _SURVEY_LANGUAGE == "eng"
            else "\nDoppelt blinzeln zum Speichern (≤ 600 ms)\n oder 'Erneut' drücken."
        )
        self.image_label.setText(f"{msg_rt}{self.reaction_time} ms{msg_save}")

        # Arm the double-blink save
        self._awaiting_save_by_dblblink = True
        self._dbl_first_ts = None

        # Do NOT show the accept/save button; show only Retry
        self.accept_reaction_button.hide()
        self.retry_button.show()

    def _poll_blinks(self) -> None:
        # Drain a few per tick
        for _ in range(4):
            ev = poll_blink()
            if ev is None:
                return

            mono, conf, remote_ts = ev
            # DEBUG:
            print(
                f"[SurveyApp] blink polled: conf={conf:.3f}, ts={remote_ts}, "
                f"awaiting_start={self.awaiting_start_blink}, "
                f"game_started={self.reactiongame_started}, "
                f"reaction_started={self.reaction_started}, "
                f"awaiting_save={self._awaiting_save_by_dblblink}"
            )

            # ---------------------------------------------------------
            # 1) Handle *double-blink-to-save* AFTER a valid reaction
            # ---------------------------------------------------------
            if self._awaiting_save_by_dblblink:
                # first blink in the pair
                if self._dbl_first_ts is None:
                    self._dbl_first_ts = remote_ts
                    try:
                        self.lsl_outlet.push_sample(
                            [f"save_first_blink,{conf:.3f},{remote_ts}"]
                        )
                    except Exception:
                        pass
                else:
                    # check time gap
                    gap_ms = (remote_ts - self._dbl_first_ts) * 1000.0
                    if gap_ms <= self._dbl_gap_ms:
                        # valid double blink → save RT
                        try:
                            self.lsl_outlet.push_sample(
                                [f"save_double_blink,{conf:.3f},{remote_ts}"]
                            )
                        except Exception:
                            pass
                        self._awaiting_save_by_dblblink = False
                        self._dbl_first_ts = None
                        self.save_reaction_time()
                        return  # done with this tick
                    else:
                        # too slow → treat this blink as first in a new pair
                        self._dbl_first_ts = remote_ts
                        try:
                            self.lsl_outlet.push_sample(
                                [f"save_first_blink_reset,{conf:.3f},{remote_ts}"]
                            )
                        except Exception:
                            pass
                # IMPORTANT: do not also use this blink for start/false/response
                continue

            # ---------------------------------------------------------
            # 2) Blink to *start* the reaction test (screen: "Blink to start")
            # ---------------------------------------------------------
            if (
                self.awaiting_start_blink
                and not self.reactiongame_started
                and not self.reaction_started
            ):
                self.awaiting_start_blink = False
                try:
                    self.lsl_outlet.push_sample([f"blink_start,{conf:.3f},{remote_ts}"])
                except Exception:
                    pass
                print("[SurveyApp] Blink START → _start_trial_after_blink()")
                self._start_trial_after_blink()
                continue

            # ---------------------------------------------------------
            # 3) Blink during RED screen → false start
            # ---------------------------------------------------------
            if self.reactiongame_started and not self.reaction_started:
                try:
                    self.lsl_outlet.push_sample([f"blink_false,{conf:.3f},{remote_ts}"])
                except Exception:
                    pass
                print("[SurveyApp] Blink FALSE START")
                self._handle_false_start()
                continue

            # ---------------------------------------------------------
            # 4) Blink during GREEN screen → valid reaction
            # ---------------------------------------------------------
            if self.reaction_started:
                try:
                    self.lsl_outlet.push_sample(
                        [f"blink_response,{conf:.3f},{remote_ts}"]
                    )
                except Exception:
                    pass
                print("[SurveyApp] Blink RESPONSE (GREEN)")
                self._reaction_blink(remote_ts)
                continue


# =============================================================================
# Qt→Pygame bridge
# =============================================================================


class QtInPygameBridge:
    """Run a hidden Qt widget while drawing it inside the CURRENT Pygame window."""

    def __init__(self) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def run_widget_blocking(self, widget: QtWidgets.QWidget) -> None:
        """Drive the widget until it emits `finished` (or window closed)."""
        widget.show()
        widget.move(-10000, -10000)  # keep Qt window offscreen
        try:
            w, h = pygame.display.get_surface().get_size()
            widget.resize(w, h)
        except Exception:
            pass

        done = {"flag": False}

        def mark_done(*_):
            done["flag"] = True

        try:
            widget.finished.connect(mark_done)  # type: ignore[attr-defined]
        except Exception:
            pass

        clock = pygame.time.Clock()
        buttons_mask = QtCore.Qt.NoButton

        while not done["flag"]:
            # Pump Qt events
            self.app.processEvents()

            # Forward Pygame events → Qt
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

            if event.type in (
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEBUTTONUP,
                pygame.MOUSEMOTION,
            ):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    qt_btn = {
                        1: QtCore.Qt.LeftButton,
                        2: QtCore.Qt.MiddleButton,
                        3: QtCore.Qt.RightButton,
                    }.get(event.button, QtCore.Qt.LeftButton)
                    buttons_mask |= qt_btn  # <-- update first
                    self._forward_mouse(widget, event, buttons_mask)

                elif event.type == pygame.MOUSEBUTTONUP:
                    qt_btn = {
                        1: QtCore.Qt.LeftButton,
                        2: QtCore.Qt.MiddleButton,
                        3: QtCore.Qt.RightButton,
                    }.get(event.button, QtCore.Qt.LeftButton)
                    buttons_mask &= ~qt_btn  # <-- update first
                    self._forward_mouse(widget, event, buttons_mask)

                else:  # pygame.MOUSEMOTION
                    # Derive current mask from real mouse state to keep it accurate during drags
                    left, middle, right = pygame.mouse.get_pressed(3)
                    mask = QtCore.Qt.NoButton
                    if left:
                        mask |= QtCore.Qt.LeftButton
                    if middle:
                        mask |= QtCore.Qt.MiddleButton
                    if right:
                        mask |= QtCore.Qt.RightButton
                    buttons_mask = mask
                    self._forward_mouse(widget, event, buttons_mask)

            # Render Qt → image → pygame surface
            disp = pygame.display.get_surface()
            if disp is None:
                clock.tick(60)
                continue
            w, h = disp.get_size()
            if widget.size() != QtCore.QSize(w, h):
                try:
                    widget.resize(w, h)
                except Exception:
                    pass

            qimg = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
            qimg.fill(QtGui.QColor(0, 0, 0, 255))
            painter = QtGui.QPainter(qimg)
            widget.render(painter)
            painter.end()

            fmt = qimg.convertToFormat(QtGui.QImage.Format_RGB888)
            ptr = fmt.bits()
            ptr.setsize(fmt.byteCount())
            surf = pygame.image.frombuffer(bytes(ptr), (w, h), "RGB")
            disp.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(60)

    @staticmethod
    def _forward_mouse(
        widget: QtWidgets.QWidget,
        event: pygame.event.Event,
        buttons_mask: QtCore.Qt.MouseButtons,
    ) -> None:
        x, y = event.pos
        target = widget.childAt(x, y) or widget
        local = target.mapFrom(widget, QtCore.QPoint(x, y))

        if event.type == pygame.MOUSEBUTTONDOWN:
            qt_btn = {
                1: QtCore.Qt.LeftButton,
                2: QtCore.Qt.MiddleButton,
                3: QtCore.Qt.RightButton,
            }.get(event.button, QtCore.Qt.LeftButton)
            qev = QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonPress,
                QtCore.QPointF(local),
                qt_btn,
                buttons_mask,  # already includes qt_btn
                QtCore.Qt.NoModifier,
            )
            QtWidgets.QApplication.sendEvent(target, qev)
            try:
                target.setFocus(QtCore.Qt.MouseFocusReason)
            except Exception:
                pass
            return

        if event.type == pygame.MOUSEBUTTONUP:
            qt_btn = {
                1: QtCore.Qt.LeftButton,
                2: QtCore.Qt.MiddleButton,
                3: QtCore.Qt.RightButton,
            }.get(event.button, QtCore.Qt.LeftButton)
            qev = QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonRelease,
                QtCore.QPointF(local),
                qt_btn,
                buttons_mask,
                QtCore.Qt.NoModifier,
            )
            QtWidgets.QApplication.sendEvent(target, qev)
            return

        # motion
        qev = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(local),
            QtCore.QPointF(local),
            QtCore.QPointF(),
            QtCore.Qt.NoButton,
            buttons_mask,
            QtCore.Qt.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(target, qev)

    @staticmethod
    def _forward_key(widget: QtWidgets.QWidget, event: pygame.event.Event) -> None:
        key, text = QtInPygameBridge._map_key(event)
        evtype = (
            QtCore.QEvent.KeyPress
            if event.type == pygame.KEYDOWN
            else QtCore.QEvent.KeyRelease
        )
        key_target = QtWidgets.QApplication.focusWidget() or widget
        qev = QtGui.QKeyEvent(evtype, key, QtCore.Qt.NoModifier, text)
        QtWidgets.QApplication.sendEvent(key_target, qev)

    @staticmethod
    def _map_key(event: pygame.event.Event) -> Tuple[int, str]:
        k = event.key
        if k == pygame.K_RETURN:
            return (QtCore.Qt.Key_Return, "")
        if k == pygame.K_BACKSPACE:
            return (QtCore.Qt.Key_Backspace, "")
        if k == pygame.K_SPACE:
            return (QtCore.Qt.Key_Space, " ")
        if pygame.K_a <= k <= pygame.K_z:
            return (
                QtCore.Qt.Key_A + (k - pygame.K_a),
                chr(ord("a") + (k - pygame.K_a)),
            )
        if pygame.K_0 <= k <= pygame.K_9:
            return (
                QtCore.Qt.Key_0 + (k - pygame.K_0),
                chr(ord("0") + (k - pygame.K_0)),
            )
        try:
            return (0, event.unicode)
        except Exception:
            return (0, "")


# =============================================================================
# Public helpers used by main
# =============================================================================


def load_survey_questions() -> List[Dict[str, Any]]:
    """Back-compat loader (unused by SurveyApp, kept if needed)."""
    path = os.path.join(_CWD_PATH, "config", "dino_config.yaml")
    with open(path, "r") as file:
        return yaml.safe_load(file).get("questions", [])


def show_survey_pyqt_and_save() -> bool:
    """Run the SurveyApp embedded in the current Pygame window."""
    bridge = QtInPygameBridge()
    try:
        part_num = int("".join([c for c in _USER_ID if c.isdigit()]))
    except Exception:
        part_num = 1
    app = SurveyApp(_CWD_PATH, part_num, cur_session=1)

    done = {"flag": False}

    def finished():
        done["flag"] = True

    app.finished.connect(finished)
    bridge.run_widget_blocking(app)
    return done["flag"]

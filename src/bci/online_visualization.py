import sys

# Check all required dependencies
missing_modules = []

try:
    import zmq
except ImportError:
    missing_modules.append("pyzmq")

try:
    import pyqtgraph as pg
except ImportError:
    missing_modules.append("pyqtgraph")

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    missing_modules.append("PyQt5")

if missing_modules:
    print("❌ ERROR: Missing required modules!")
    print(f"   Missing: {', '.join(missing_modules)}")
    print("\n   Install them in your conda environment:")
    print(f"   conda activate bci-challenge")
    print(f"   pip install {' '.join(missing_modules)}")
    print("\n   Or install all at once:")
    print(
        f"   /opt/miniconda3/envs/bci-challenge/bin/pip install {' '.join(missing_modules)}"
    )
    sys.exit(1)

from collections import deque
from pathlib import Path

import numpy as np

from bci.utils.bci_config import load_config

# --- Configuration ---

root = Path.cwd()
# Setup Config
config = load_config(root / "resources" / "configs" / "bci_config.yaml")

PORT = "5556"
WINDOW_SIZE = 1000  # Total samples visible on screen
REFRESH_RATE_MS = 20  # 50 FPS

# Offset between channels on the plot (visual separation)
# Since we scale Volts to uV (1e6), 50 means 50uV separation.
DEFAULT_Y_OFFSET = 50

# Fallback names
DEFAULT_CHANNEL_NAMES = [
    ch
    for ch in config.channels
    if not config.remove_channels or ch not in config.remove_channels
]

THRESHOLD = config.classification_threshold


class ZMQListener(QtCore.QThread):
    packet_received = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://localhost:{PORT}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"Visualization Listener started on port {PORT}...")

        while self.running:
            try:
                # Blocking receive
                msg = socket.recv_pyobj()
                self.packet_received.emit(msg)
            except zmq.ZMQError:
                break
            except Exception as e:
                print(f"ZMQ Error: {e}")
                break

        socket.close()
        context.term()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class EEGVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-time BCI Monitor + Robust Probabilities")
        self.resize(1200, 900)

        # --- State ---
        self.n_channels = 0
        self.channel_names = []
        self.initialized = False

        # Buffers
        self.buffer_raw = None
        self.buffer_filt = None
        self.ptr = 0

        # Scaling: 1e6 converts Volts to uV
        self.current_scale = 1000000.0

        # Data Queue
        self.data_queue = deque()
        self.current_chunk_raw = None
        self.current_chunk_filt = None
        self.current_chunk_idx = 0

        # Visual Objects
        self.curves = []
        self.markers = []
        self.proba_bar_item = None  # The bar chart object

        # --- UI Init ---
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        self.sidebar_widget = QtWidgets.QWidget()
        self.graph_widget = pg.GraphicsLayoutWidget()

        self._setup_skeleton_ui()

        # --- Logic Init ---
        self.thread = ZMQListener()
        self.thread.packet_received.connect(self.queue_data)
        self.thread.start()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(REFRESH_RATE_MS)
        self.timer.timeout.connect(self.update_loop)
        self.timer.start()

    def _setup_skeleton_ui(self):
        """Creates the empty containers, waiting for data."""
        # Sidebar
        self.sidebar_widget.setFixedWidth(250)
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar_widget)
        self.main_layout.addWidget(self.sidebar_widget)

        # Graph Area
        self.main_layout.addWidget(self.graph_widget)

        # Loading Label
        # Use addLabel instead of addItem(QLabel) to avoid crashes
        self.loading_label = self.graph_widget.addLabel(
            "Waiting for Data Stream...", size="18pt", color="#888"
        )

    def _initialize_buffers(self, n_channels):
        """Called once the first packet arrives."""
        self.n_channels = n_channels
        self.buffer_raw = np.zeros((n_channels, WINDOW_SIZE))
        self.buffer_filt = np.zeros((n_channels, WINDOW_SIZE))

        if n_channels <= len(DEFAULT_CHANNEL_NAMES):
            self.channel_names = DEFAULT_CHANNEL_NAMES[:n_channels]
        else:
            self.channel_names = [f"Ch {i+1}" for i in range(n_channels)]

        # --- Rebuild UI ---
        self.graph_widget.clear()  # Removes the "Waiting..." label

        # --- 1. Top Plot: EEG Signals ---
        self.plot = self.graph_widget.addPlot(title="Live EEG Feed")
        self.plot.setXRange(0, WINDOW_SIZE)

        # Set Y range to cover all channels with spacing
        total_height = (n_channels * DEFAULT_Y_OFFSET) + DEFAULT_Y_OFFSET
        self.plot.setYRange(-DEFAULT_Y_OFFSET, total_height)

        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Amplitude", units="uV")

        # Create Curves
        for i in range(n_channels):
            color = pg.intColor(i, hues=n_channels, values=1, maxValue=255, alpha=200)
            c = self.plot.plot(pen=pg.mkPen(color, width=1.5))
            self.curves.append(c)

        # Custom Y-Axis Ticks
        axis = self.plot.getAxis("left")
        ticks = [
            (i * DEFAULT_Y_OFFSET, self.channel_names[i]) for i in range(n_channels)
        ]
        axis.setTicks([ticks])

        # Sweep Line (Green)
        self.vline = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("g", width=2)
        )
        self.plot.addItem(self.vline)

        # --- 2. Bottom Plot: Probabilities ---
        self.graph_widget.nextRow()  # Move grid cursor to next row

        self.proba_plot = self.graph_widget.addPlot(
            title="Class Probabilities (Robust Buffer)"
        )
        self.proba_plot.setMaximumHeight(200)  # Keep it shorter than EEG
        self.proba_plot.setYRange(0, 1.05)
        self.proba_plot.setXRange(-0.5, 2.5)

        # X-Axis Labels for Classes
        # Assuming classes are: 0=Rest, 1=Left, 2=Right
        ax_bottom = self.proba_plot.getAxis("bottom")
        ax_bottom.setTicks([[(0, "Rest"), (1, "Left"), (2, "Right")]])

        # Bar Chart Item
        # Initial heights are 0
        self.proba_bar_item = pg.BarGraphItem(
            x=[0, 1, 2],
            height=[0, 0, 0],
            width=0.6,
            brush="#3F51B5",  # Indigo color
        )
        self.proba_plot.addItem(self.proba_bar_item)

        # Threshold Line (0.6)
        thresh_line = pg.InfiniteLine(
            pos=THRESHOLD,
            angle=0,
            movable=False,
            pen=pg.mkPen("r", width=2, style=QtCore.Qt.DashLine),
            label=f"Threshold ({THRESHOLD})",
            labelOpts={"position": 0.05, "color": (200, 50, 50)},
        )
        self.proba_plot.addItem(thresh_line)

        # --- 3. Controls ---
        self._build_sidebar_controls()

        self.initialized = True
        print(f"Initialized with {n_channels} channels.")

    def _build_sidebar_controls(self):
        gb = QtWidgets.QGroupBox("Display Settings")
        l = QtWidgets.QVBoxLayout()

        self.cb_show_filtered = QtWidgets.QCheckBox("Show Filtered Signal")
        self.cb_show_filtered.setChecked(True)
        l.addWidget(self.cb_show_filtered)

        l.addWidget(QtWidgets.QLabel("Vertical Scale (Gain):"))
        self.spin_scale = QtWidgets.QDoubleSpinBox()
        # Allow huge numbers for Volts input
        self.spin_scale.setRange(1.0, 1000000000.0)
        self.spin_scale.setSingleStep(10000.0)
        self.spin_scale.setValue(self.current_scale)
        self.spin_scale.valueChanged.connect(
            lambda v: setattr(self, "current_scale", v)
        )
        l.addWidget(self.spin_scale)

        btn_auto = QtWidgets.QPushButton("Auto Scale")
        btn_auto.setStyleSheet("background-color: #2E7D32; color: white;")
        btn_auto.clicked.connect(self.perform_auto_scale)
        l.addWidget(btn_auto)

        gb.setLayout(l)
        self.sidebar_layout.addWidget(gb)

        # Channels List
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        self.clayout = QtWidgets.QVBoxLayout(content)

        for i, name in enumerate(self.channel_names):
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel()
            lbl.setFixedSize(14, 14)
            color = pg.intColor(i, hues=self.n_channels, values=1, maxValue=255).name()
            lbl.setStyleSheet(f"background-color: {color}; border-radius: 7px;")

            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(
                lambda s, idx=i: self.curves[idx].setVisible(s == QtCore.Qt.Checked)
            )

            row.addWidget(lbl)
            row.addWidget(cb)
            row.addStretch()
            self.clayout.addLayout(row)

        self.clayout.addStretch()
        scroll.setWidget(content)
        self.sidebar_layout.addWidget(scroll)

    def queue_data(self, packet):
        self.data_queue.append(packet)

    def update_loop(self):
        """Processes data from the queue and updates the display."""
        if not self.data_queue:
            return

        samples_processed = 0
        MAX_SAMPLES = 100

        while self.data_queue and samples_processed < MAX_SAMPLES:
            # Fetch next item if needed
            if self.current_chunk_raw is None:
                packet = self.data_queue.popleft()
                tag = packet[0]

                if tag == "PROBA":
                    # --- Update Bar Chart ---
                    # packet[1] is [p_rest, p_left, p_right]
                    if self.proba_bar_item is not None:
                        probs = packet[1]
                        # Ensure we don't crash if dims mismatch
                        if len(probs) == 3:
                            self.proba_bar_item.setOpts(height=probs)
                    continue

                elif tag == "EVENT":
                    self.add_marker(packet[1])
                    continue

                elif tag == "DATA":
                    raw, filt = packet[1], packet[2]

                    if not self.initialized:
                        self._initialize_buffers(raw.shape[0])

                    self.current_chunk_raw = raw
                    self.current_chunk_filt = filt
                    self.current_chunk_idx = 0

            if self.current_chunk_raw is None:
                continue

            # Process Chunk Data
            chunk_len = self.current_chunk_raw.shape[1]
            remaining = chunk_len - self.current_chunk_idx

            # Write all remaining data in this chunk
            start = self.current_chunk_idx
            end = chunk_len
            n = end - start

            self._write_to_buffer(
                self.current_chunk_raw[:, start:end],
                self.current_chunk_filt[:, start:end],
            )

            self.current_chunk_raw = None
            samples_processed += n

        if self.initialized and samples_processed > 0:
            self._redraw()

    def _write_to_buffer(self, raw, filt):
        n = raw.shape[1]
        self._clean_markers(self.ptr, n)

        if self.ptr + n <= WINDOW_SIZE:
            self.buffer_raw[:, self.ptr : self.ptr + n] = raw
            self.buffer_filt[:, self.ptr : self.ptr + n] = filt
            self.ptr += n
        else:
            overflow = (self.ptr + n) - WINDOW_SIZE
            self.buffer_raw[:, self.ptr :] = raw[:, :-overflow]
            self.buffer_filt[:, self.ptr :] = filt[:, :-overflow]
            self.buffer_raw[:, :overflow] = raw[:, -overflow:]
            self.buffer_filt[:, :overflow] = filt[:, -overflow:]
            self.ptr = overflow
            self._clean_markers(0, overflow)

    def _clean_markers(self, start, length):
        end = start + length
        for i in range(len(self.markers) - 1, -1, -1):
            if start <= self.markers[i]["pos"] < end:
                self.plot.removeItem(self.markers[i]["line"])
                self.markers.pop(i)

    def add_marker(self, label):
        if not self.initialized:
            return

        color = "r" if "Artifact" in str(label) else "00FFFF"
        style = QtCore.Qt.DashLine if "Artifact" in str(label) else QtCore.Qt.SolidLine

        line = pg.InfiniteLine(
            pos=self.ptr,
            angle=90,
            movable=False,
            pen=pg.mkPen(color, width=2, style=style),
            label=str(label),
            labelOpts={
                "position": 0.9,
                "color": (200, 200, 200),
                "movable": True,
                "fill": (0, 0, 0, 100),
            },
        )
        self.plot.addItem(line)
        self.markers.append({"line": line, "pos": self.ptr})

    def _redraw(self):
        self.vline.setPos(self.ptr)

        # Select buffer based on checkbox
        data = (
            self.buffer_filt if self.cb_show_filtered.isChecked() else self.buffer_raw
        )

        for i in range(self.n_channels):
            if self.curves[i].isVisible():
                channel_data = data[i]

                # --- STEP 1: DC OFFSET REMOVAL ---
                # Calculate the average voltage of the currently visible window
                # and subtract it. This centers the signal at 0.
                center_bias = np.mean(channel_data)
                centered_signal = channel_data - center_bias

                # --- STEP 2: SCALING & POSITIONING ---
                # Now we scale the pure AC signal and add the channel's vertical offset
                trace = (centered_signal * self.current_scale) + (i * DEFAULT_Y_OFFSET)

                self.curves[i].setData(trace)

    def perform_auto_scale(self):
        if self.buffer_filt is None:
            return
        data = (
            self.buffer_filt if self.cb_show_filtered.isChecked() else self.buffer_raw
        )

        # Use first channel std to estimate scale
        std = np.std(data[0])
        if std < 1e-15:
            return

        # Fit 3 sigmas into 80% of the channel gap
        new_scale = (DEFAULT_Y_OFFSET * 0.8) / (6 * std)
        self.spin_scale.setValue(new_scale)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Dark Theme
    app.setStyle("Fusion")
    p = QtGui.QPalette()
    p.setColor(QtGui.QPalette.Window, QtGui.QColor(40, 40, 40))
    p.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    p.setColor(QtGui.QPalette.Base, QtGui.QColor(20, 20, 20))
    p.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    p.setColor(QtGui.QPalette.Button, QtGui.QColor(40, 40, 40))
    p.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    p.setColor(QtGui.QPalette.Highlight, QtGui.QColor(46, 125, 50))
    app.setPalette(p)

    viz = EEGVisualizer()
    viz.show()
    sys.exit(app.exec_())

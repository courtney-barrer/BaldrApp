"""Editable NCPA preview; slider movement never sends an apply command."""
import json
import uuid
import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from baldrapp.apps.paranal_simulator.sim_ncpa import MODE_NAMES, make_ncpa_basis, compose_ncpa


class NcpaDialog(QtWidgets.QDialog):
    def __init__(self, send_command, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NCPA — preview and apply")
        self.resize(1050, 740)
        self.send_command = send_command
        self.pending = None
        self.profile = None
        self.applied = None
        layout = QtWidgets.QVBoxLayout(self)
        selection = QtWidgets.QHBoxLayout()
        self.beam = QtWidgets.QComboBox()
        self.beam.addItems(["1", "2", "3", "4"])
        self.source = QtWidgets.QComboBox()
        self.source.addItems(["onsky","internal"])
        selection.addWidget(QtWidgets.QLabel("Beam"))
        selection.addWidget(self.beam)
        selection.addWidget(QtWidgets.QLabel("Source profile"))
        selection.addWidget(self.source)
        self.reload = QtWidgets.QPushButton("Reload applied")
        selection.addWidget(self.reload)
        layout.addLayout(selection)
        layout.addWidget(QtWidgets.QLabel(
            "Signed amplitudes: nm RMS OPD per mode. Profiles are retained until simulator restart.\n"
            "Switching beam/source reloads its applied settings and discards unapplied edits."
        ))
        body = QtWidgets.QHBoxLayout()
        self.editors = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(self.editors)
        self.spins = []
        for row, name in enumerate(MODE_NAMES):
            grid.addWidget(QtWidgets.QLabel(name), row, 0)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(-1000, 1000)
            slider.setToolTip("Slider: ±1000 nm. Numeric entry: ±10000 nm.")
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-10000, 10000)
            spin.setDecimals(2)
            spin.setSuffix(" nm")
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(lambda value, s=slider: self.sync_slider(s, value))
            spin.valueChanged.connect(self.schedule_preview)
            grid.addWidget(slider, row, 1)
            grid.addWidget(spin, row, 2)
            self.spins.append(spin)
        body.addWidget(self.editors)
        figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvasQTAgg(figure)
        self.axes = figure.add_subplot(111)
        self.map_artist = None
        body.addWidget(self.canvas)
        layout.addLayout(body)
        self.combined = QtWidgets.QCheckBox("Preview combined INTERNAL_OPD (configured aberrations + NCPA)")
        self.combined.toggled.connect(self.schedule_preview)
        layout.addWidget(self.combined)
        self.status = QtWidgets.QLabel("Loading profile…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QHBoxLayout()
        self.reset = QtWidgets.QPushButton("Reset coefficients")
        self.apply = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel / Close")
        for button in (self.reset, self.apply, cancel):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.update_preview)
        self.reset.clicked.connect(self.reset_coefficients)
        self.apply.clicked.connect(self.apply_coefficients)
        cancel.clicked.connect(self.reject)
        self.reload.clicked.connect(self.load_profile)
        self.beam.currentIndexChanged.connect(self.load_profile)
        self.source.currentIndexChanged.connect(self.load_profile)
        self.set_busy(True)

    @staticmethod
    def sync_slider(slider, value):
        blocker = QtCore.QSignalBlocker(slider)
        slider.setValue(round(value))
        del blocker

    def set_busy(self, busy):
        for widget in (self.beam, self.source, self.reload):
            widget.setEnabled(not busy)
        for widget in (self.editors, self.apply, self.reset):
            widget.setEnabled(not busy and self.profile is not None)

    def request(self, action, **extra):
        data = dict(beam=int(self.beam.currentText()), source=self.source.currentText(),
                    request_id=uuid.uuid4().hex, **extra)
        self.pending = "ncpa " + action + " " + json.dumps(data)
        self.set_busy(True)
        self.send_command(self.pending)

    def load_profile(self, *_):
        self.profile = None
        self.applied = None
        self.status.setText("Loading profile…")
        self.axes.set_title("Loading selected pupil…")
        if self.map_artist is not None:
            self.map_artist.set_visible(False)
        self.canvas.draw_idle()
        self.request("get")

    def handle_reply(self, command, payload, transport_ok):
        if command != self.pending:
            return
        self.pending = None
        try:
            if not transport_ok or not payload.get("ok"):
                raise ValueError(payload.get("message", "No reply from simulator; reload to check applied state."))
            data = payload["ncpa"]
            if command.startswith("ncpa get "):
                pupil = np.asarray(data["pupil"], dtype=float)
                self.profile = dict(data, pupil=pupil, baseline=np.asarray(data["baseline"]),
                                    basis=make_ncpa_basis(pupil, data["diameter"]))
                for spin, value in zip(self.spins, data["coefficients_nm"]):
                    spin.setValue(value)
            self.applied = np.asarray(data["coefficients_nm"])
            self.profile["revision"] = data["revision"]
            self.update_preview()
        except (ValueError, KeyError, TypeError) as exc:
            self.status.setText(str(exc))
        finally:
            self.set_busy(False)

    def coefficients(self):
        return [spin.value() for spin in self.spins]

    def schedule_preview(self, *_):
        self.timer.start()

    def update_preview(self):
        if self.profile is None:
            return
        values = self.coefficients()
        opd = compose_ncpa(self.profile["basis"], values)
        if self.combined.isChecked():
            opd = opd + self.profile["baseline"]
        mask = self.profile["pupil"] > 0
        display = np.ma.array(opd*1e9, mask=~mask)
        limit = max(float(np.max(np.abs(display))), 1.0)
        if self.map_artist is None:
            self.map_artist = self.axes.imshow(display, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
            self.canvas.figure.colorbar(self.map_artist, ax=self.axes, label="OPD (nm)")
            self.axes.set_xlabel("OPD column (includes padding)")
            self.axes.set_ylabel("OPD row (includes padding)")
        else:
            self.map_artist.set_visible(True)
            self.map_artist.set_data(display)
            h, w = mask.shape
            self.map_artist.set_extent((-0.5, w-0.5, -0.5, h-0.5))
            self.axes.set_xlim(-0.5, w-0.5)
            self.axes.set_ylim(-0.5, h-0.5)
            self.map_artist.set_clim(-limit, limit)
        title = "Combined INTERNAL_OPD" if self.combined.isChecked() else "Adjustable NCPA"
        self.axes.set_title(title + f" — beam {self.beam.currentText()}, {self.source.currentText()}")
        rms = np.std(opd[mask])*1e9
        dirty = self.applied is None or not np.array_equal(values, self.applied)
        state = "Unapplied edits" if dirty else f"Applied profile revision {self.profile['revision']}"
        self.status.setText(f"{state}. Preview RMS (piston removed): {rms:.2f} nm. "
                            "The profile is used on subsequent frames when its source is active.")
        self.canvas.draw_idle()

    def reset_coefficients(self):
        for spin in self.spins:
            spin.setValue(0)

    def apply_coefficients(self):
        self.timer.stop()
        self.status.setText("Applying profile…")
        self.request("set", coefficients_nm=self.coefficients())

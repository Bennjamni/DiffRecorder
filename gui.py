"""Interface gráfica moderna para o gravador de áudio.

Requisitos adicionais: PySide6

Funcionalidades:
- Lista dispositivos de entrada
- Escolhe taxa (48 kHz / 192 kHz)
- Escolhe canais e duração ou gravação contínua
- Botões Gravar / Parar
- Salva WAV em PCM_24
"""

import sys
import threading
import time
import collections
import numpy as np
from datetime import datetime

import sounddevice as sd
import soundfile as sf

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QFileDialog,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QLineEdit,
)
from PySide6.QtCore import Qt, Signal, QObject


from PySide6.QtGui import QPainter, QColor, QPen, QPaintEvent


class WaveformWidget(QWidget):
    """Widget simples para desenhar a waveform em tempo real.

    Mantém um buffer circular dos últimos segundos de áudio e desenha uma linha
    representando a forma de onda (mix mono se multi-canal).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.samplerate = 48000
        self.display_seconds = 5.0
        self.max_samples = int(self.samplerate * self.display_seconds)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self._lock = threading.Lock()

    def set_samplerate(self, sr):
        with self._lock:
            self.samplerate = sr
            self.max_samples = int(self.samplerate * self.display_seconds)
            old = list(self.buffer)
            self.buffer = collections.deque(old[-self.max_samples:], maxlen=self.max_samples)

    def append_chunk(self, chunk: np.ndarray):
        # chunk shape: (frames, channels) or (frames,)
        if chunk is None:
            return
        if chunk.ndim == 1:
            mono = chunk
        else:
            # mix to mono
            mono = np.mean(chunk, axis=1)

        with self._lock:
            # extend buffer
            for v in mono:
                self.buffer.append(float(v))
        # ask for repaint
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        rect = self.rect()
        # background
        painter.fillRect(rect, QColor('#0f1112'))
        # draw center line
        pen = QPen(QColor('#2b2d2f'))
        pen.setWidth(1)
        painter.setPen(pen)
        midy = rect.height() // 2
        painter.drawLine(0, midy, rect.width(), midy)

        # draw waveform
        with self._lock:
            buf = np.array(self.buffer, dtype=np.float32)
        if buf.size == 0:
            return

        # downsample/scale to widget width
        w = rect.width()
        if buf.size < w:
            xs = np.linspace(0, w - 1, buf.size).astype(int)
            samples = buf
        else:
            # compute envelope per pixel column
            factor = int(np.ceil(buf.size / float(w)))
            samples = buf[:factor * w].reshape(w, factor)
            samples = samples.mean(axis=1)
            xs = np.arange(w)

        # normalize
        max_val = max(1e-6, np.max(np.abs(samples)))
        ys = (samples / max_val) * (rect.height() / 2 - 4)

        pen = QPen(QColor('#3aa0ff'))
        pen.setWidth(1)
        painter.setPen(pen)

        prev_x = 0
        prev_y = int(midy - ys[0])
        for i, val in enumerate(ys):
            x = int(i * (rect.width() / max(1, len(ys))))
            y = int(midy - val)
            painter.drawLine(prev_x, prev_y, x, y)
            prev_x, prev_y = x, y



class RecorderWorker(QObject):
    status = Signal(str)
    finished = Signal(str)
    error = Signal(str)
    data = Signal(object)

    def __init__(self, device, samplerate, channels, duration, filename, blocksize=1024,
                 enable_hpf=False, hpf_cutoff=80.0, gate_threshold=0.0):
        super().__init__()
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.duration = duration
        self.filename = filename
        self.blocksize = blocksize
        self._stop = threading.Event()
        # high-pass filter state
        self.enable_hpf = enable_hpf
        self.hpf_cutoff = float(hpf_cutoff)
        self._hpf_x1 = None
        self._hpf_y1 = None
        # noise gate
        self.gate_threshold = float(gate_threshold)

    def stop(self):
        self._stop.set()

    def run(self):
        self.status.emit("Iniciando gravação...")
        subtype = 'PCM_24'
        start_time = time.time()
        try:
            with sf.SoundFile(self.filename, mode='w', samplerate=self.samplerate,
                              channels=self.channels, subtype=subtype) as file:
                with sd.InputStream(samplerate=self.samplerate, device=self.device,
                                     channels=self.channels, dtype='float32') as stream:
                    self.status.emit("Gravando — pressione Parar para encerrar")
                    while not self._stop.is_set():
                        data, _ = stream.read(self.blocksize)

                        # Apply simple high-pass filter per channel (one-pole IIR using difference)
                        if self.enable_hpf and self.hpf_cutoff > 0.0:
                            # prepare state
                            if self._hpf_x1 is None:
                                # initialize previous samples and outputs
                                if data.ndim == 1:
                                    ch = 1
                                else:
                                    ch = data.shape[1]
                                self._hpf_x1 = np.zeros(ch, dtype=np.float32)
                                self._hpf_y1 = np.zeros(ch, dtype=np.float32)

                            # filter coefficients (simple RC high-pass)
                            dt = 1.0 / float(self.samplerate)
                            rc = 1.0 / (2.0 * np.pi * float(self.hpf_cutoff))
                            alpha = rc / (rc + dt)

                            if data.ndim == 1:
                                x = data
                                y = np.empty_like(x)
                                x1 = self._hpf_x1[0]
                                y1 = self._hpf_y1[0]
                                for i, xi in enumerate(x):
                                    yi = alpha * (y1 + xi - x1)
                                    y[i] = yi
                                    x1 = xi
                                    y1 = yi
                                self._hpf_x1[0] = x1
                                self._hpf_y1[0] = y1
                                data = y
                            else:
                                # per-channel
                                y = np.empty_like(data)
                                for ch in range(data.shape[1]):
                                    x = data[:, ch]
                                    x1 = self._hpf_x1[ch]
                                    y1 = self._hpf_y1[ch]
                                    for i, xi in enumerate(x):
                                        yi = alpha * (y1 + xi - x1)
                                        y[i, ch] = yi
                                        x1 = xi
                                        y1 = yi
                                    self._hpf_x1[ch] = x1
                                    self._hpf_y1[ch] = y1
                                data = y

                        # Apply simple noise gate if requested
                        if self.gate_threshold and self.gate_threshold > 0.0:
                            if data.ndim == 1:
                                mask = np.abs(data) < self.gate_threshold
                                data[mask] = 0.0
                            else:
                                mag = np.mean(np.abs(data), axis=1)
                                mask = mag < self.gate_threshold
                                data[mask, :] = 0.0

                        file.write(data)
                        try:
                            # Emit a copy of the data for UI waveform (may be multi-channel)
                            self.data.emit(np.copy(data))
                        except Exception:
                            # If waveform update fails, continue recording
                            pass
                        if self.duration is not None:
                            if time.time() - start_time >= self.duration:
                                break
            self.finished.emit(self.filename)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DiffRecorder — Gravador")
        self.setMinimumSize(560, 320)
        self.worker = None
        self.worker_thread = None

        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout()

        # Device selection
        h_dev = QHBoxLayout()
        h_dev.addWidget(QLabel("Dispositivo de entrada:"))
        self.device_combo = QComboBox()
        h_dev.addWidget(self.device_combo)
        layout.addLayout(h_dev)

        # Sample rate
        h_sr = QHBoxLayout()
        h_sr.addWidget(QLabel("Taxa de amostragem:"))
        self.sr_group = QButtonGroup()
        self.sr48 = QRadioButton("48 kHz (24-bit)")
        self.sr192 = QRadioButton("192 kHz (24-bit)")
        self.sr48.setChecked(True)
        self.sr_group.addButton(self.sr48)
        self.sr_group.addButton(self.sr192)
        h_sr.addWidget(self.sr48)
        h_sr.addWidget(self.sr192)
        layout.addLayout(h_sr)

        # Channels and duration
        h_opts = QHBoxLayout()
        h_opts.addWidget(QLabel("Canais:"))
        self.channels_spin = QSpinBox()
        self.channels_spin.setMinimum(1)
        self.channels_spin.setMaximum(32)
        h_opts.addWidget(self.channels_spin)

        h_opts.addWidget(QLabel("Duração (s):"))
        self.duration_edit = QLineEdit()
        self.duration_edit.setPlaceholderText("vazio = gravação contínua")
        h_opts.addWidget(self.duration_edit)

        layout.addLayout(h_opts)

        # Noise reduction options
        h_noise = QHBoxLayout()
        self.hpf_checkbox = QCheckBox("Ativar high-pass filter (remove rumble)")
        self.hpf_cut_spin = QSpinBox()
        self.hpf_cut_spin.setRange(10, 1000)
        self.hpf_cut_spin.setValue(80)
        h_noise.addWidget(self.hpf_checkbox)
        h_noise.addWidget(QLabel("Corte (Hz):"))
        h_noise.addWidget(self.hpf_cut_spin)
        layout.addLayout(h_noise)

        # File selection
        h_file = QHBoxLayout()
        h_file.addWidget(QLabel("Arquivo de saída:"))
        self.file_edit = QLineEdit()
        h_file.addWidget(self.file_edit)
        self.browse_btn = QPushButton("Escolher")
        self.browse_btn.clicked.connect(self.choose_file)
        h_file.addWidget(self.browse_btn)
        layout.addLayout(h_file)

        # Buttons
        h_btn = QHBoxLayout()
        self.record_btn = QPushButton("Gravar")
        self.stop_btn = QPushButton("Parar")
        self.stop_btn.setEnabled(False)
        h_btn.addWidget(self.record_btn)
        h_btn.addWidget(self.stop_btn)
        layout.addLayout(h_btn)

        # Status
        self.status_label = QLabel("Pronto")
        layout.addWidget(self.status_label)

        # Waveform display
        self.waveform = WaveformWidget()
        self.waveform.setMinimumHeight(140)
        layout.addWidget(self.waveform)

        central.setLayout(layout)
        self.setCentralWidget(central)

        # populate devices
        self._load_devices()

        # signals
        self.record_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget { background: #111214; color: #e6eef6; font-family: 'Segoe UI', Tahoma, sans-serif; }
            QLineEdit, QComboBox, QSpinBox { background: #151617; border: 1px solid #2b2d2f; padding: 6px; }
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #3aa0ff, stop:1 #0066cc); border-radius: 8px; padding: 8px 14px; }
            QPushButton:disabled { background: #444546; }
            QLabel { font-size: 12pt; }
            QRadioButton { padding: 4px; }
        """)

    def _load_devices(self):
        self.device_combo.clear()
        try:
            devices = sd.query_devices()
            inputs = []
            for i, d in enumerate(devices):
                if d.get('max_input_channels', 0) > 0:
                    label = f"{i}: {d['name']} (ch={d['max_input_channels']})"
                    self.device_combo.addItem(label, i)
                    inputs.append((i, d))
            # set sensible defaults
            if inputs:
                idx, d = inputs[0]
                self.channels_spin.setValue(min(2, d.get('max_input_channels', 1)))
        except Exception as e:
            self.status_label.setText(f"Erro listando dispositivos: {e}")

    def choose_file(self):
        default_name = datetime.now().strftime('recording_%Y%m%d_%H%M%S.wav')
        path, _ = QFileDialog.getSaveFileName(self, "Salvar como", default_name, "Wave files (*.wav)")
        if path:
            self.file_edit.setText(path)

    def start_recording(self):
        # gather settings
        idx = self.device_combo.currentData()
        if idx is None:
            self.status_label.setText("Selecione um dispositivo antes de gravar.")
            return
        samplerate = 48000 if self.sr48.isChecked() else 192000
        channels = int(self.channels_spin.value())
        dur_text = self.duration_edit.text().strip()
        duration = None
        if dur_text != "":
            try:
                duration = float(dur_text)
                if duration <= 0:
                    raise ValueError()
            except ValueError:
                self.status_label.setText("Duração inválida. Use número de segundos ou deixe vazio.")
                return

        filename = self.file_edit.text().strip()
        if not filename:
            # default name in cwd
            filename = datetime.now().strftime('recording_%Y%m%d_%H%M%S.wav')
            self.file_edit.setText(filename)

        # disable inputs
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.device_combo.setEnabled(False)
        self.sr48.setEnabled(False)
        self.sr192.setEnabled(False)
        self.channels_spin.setEnabled(False)
        self.duration_edit.setEnabled(False)
        self.browse_btn.setEnabled(False)

        # create worker
        self.worker = RecorderWorker(device=idx, samplerate=samplerate, channels=channels,
                                     duration=duration, filename=filename,
                                     enable_hpf=self.hpf_checkbox.isChecked(),
                                     hpf_cutoff=self.hpf_cut_spin.value())
        self.worker.status.connect(self._on_status)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.data.connect(self.waveform.append_chunk)

        # run in Python thread
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        # ensure waveform uses correct samplerate
        try:
            self.waveform.set_samplerate(samplerate)
        except Exception:
            pass
        self.worker_thread.start()
        self._on_status("Gravando...")

    def stop_recording(self):
        if self.worker:
            self.worker.stop()
            self._on_status("Parando...")
            # worker will emit finished when done

    def _on_status(self, text):
        self.status_label.setText(text)

    def _on_finished(self, filename):
        self.status_label.setText(f"Gravação salva: {filename}")
        self._reset_ui()

    def _on_error(self, text):
        self.status_label.setText(f"Erro: {text}")
        self._reset_ui()

    def _reset_ui(self):
        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.device_combo.setEnabled(True)
        self.sr48.setEnabled(True)
        self.sr192.setEnabled(True)
        self.channels_spin.setEnabled(True)
        self.duration_edit.setEnabled(True)
        self.browse_btn.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

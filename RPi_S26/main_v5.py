import sys
import queue
from pathlib import Path

import numpy as np
import pandas as pd
import os
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QComboBox, QVBoxLayout, QHBoxLayout, QLabel,
    QDesktopWidget, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QFontMetrics, QPainter, QPainterPath, QPen, QColor, QIcon
from PyQt5.QtCore import QRectF

from models.knn_hindi.knn_hindi import KNN_Hindi_model

# =============================================================================
# USER-CONFIGURABLE CONSTANTS
# =============================================================================
SAMPLING_RATE      = 256          # Hz
WINDOW_SEC         = 5            # seconds per inference segment
WINDOW_SAMPLES     = SAMPLING_RATE * WINDOW_SEC   # 1280

ROLLING_SEC        = 20           # seconds of history shown in rolling plot
ROLLING_SAMPLES    = SAMPLING_RATE * ROLLING_SEC  # 5120

PLOT_DOWNSAMPLE    = 32           # plot every Nth sample  â†’  256/32 = 8 updates/sec
PLOT_INTERVAL_MS   = int(1000 / (SAMPLING_RATE / PLOT_DOWNSAMPLE))  # ~125 ms

DISPLAY_CHANNELS   = ['EEG.Cz', 'EEG.Fz', 'EEG.C3', 'EEG.C4', 'EEG.Pz', 'EEG.Oz']
CHANNEL_COLORS     = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

Y_MIN, Y_MAX       = -4.0, 4.0   # fixed y-axis range (data is z-score normalized)

ALL_HINDI_CHANNELS = [
    'EEG.Cz',  'EEG.Fz',  'EEG.Fp1', 'EEG.F7',  'EEG.F3',  'EEG.FC1',
    'EEG.C3',  'EEG.FC5', 'EEG.FT9', 'EEG.T7',  'EEG.TP9', 'EEG.CP5',
    'EEG.CP1', 'EEG.P3',  'EEG.P7',  'EEG.O1',  'EEG.Pz',  'EEG.Oz',
    'EEG.O2',  'EEG.P8',  'EEG.P4',  'EEG.CP2', 'EEG.CP6', 'EEG.TP10',
    'EEG.T8',  'EEG.FT10','EEG.FC6', 'EEG.C4',  'EEG.FC2', 'EEG.F4',
    'EEG.F8',  'EEG.Fp2'
]

# indices of DISPLAY_CHANNELS within ALL_HINDI_CHANNELS
DISPLAY_INDICES = [ALL_HINDI_CHANNELS.index(ch) for ch in DISPLAY_CHANNELS]


# =============================================================================
# Shared stylesheet
# =============================================================================
STYLESHEET = """
    QWidget {
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
            stop:0 #eaf6ff, stop:1 #b3e0ff);
        font-family: 'Segoe UI', sans-serif;
    }
    QLabel { font-size: 16px; color: #333; }
    QComboBox {
        font-size: 20px;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: bold;
        background-color: #77b9f7;
        color: black;
        border: 2px solid #000000;
    }
    QComboBox:hover { background-color: #45a1f7; }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 24px;
        border-left: 2px solid black;
    }
    QComboBox::down-arrow {
        image: url(utils/arrow.png);
        width: 12px; height: 12px;
    }
    QPushButton {
        font-size: 20px;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: bold;
        border: 2px solid #000000;
        background-color: #77b9f7;
        color: black;
    }
    QPushButton:hover { background-color: #45a1f7; }
"""


# =============================================================================
# Custom rounded label
# =============================================================================
class CustomLabel(QLabel):
    def __init__(self, *args, padding=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = padding
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(
            self.padding // 2, self.padding // 2,
            -self.padding // 2, -self.padding // 2
        )
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 20, 20)
        painter.setPen(QPen(QColor("#000000"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        super().paintEvent(event)


# =============================================================================
# Inference worker thread
# =============================================================================
class InferenceWorker(QThread):
    result_ready = pyqtSignal(int, int, int, str)   # seg_idx, start_sec, end_sec, word

    def __init__(self, model, seg_queue):
        super().__init__()
        self.model     = model
        self.seg_queue = seg_queue
        self._running  = True

    def run(self):
        while self._running:
            try:
                seg_idx, start_sec, end_sec, window = self.seg_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                batch  = window[np.newaxis, :, :]          # (1, 1280, 32)
                epochs = self.model._preprocess(batch)     # (n_epochs, 256, 32)
                pred   = self.model.predict(epochs)
                word   = "PAIN" if pred == 0 else "LIGHT"
            except Exception as e:
                word = f"ERR:{e}"
            self.result_ready.emit(seg_idx, start_sec, end_sec, word)
            self.seg_queue.task_done()

    def stop(self):
        self._running = False


# =============================================================================
# Child window â€” rolling EEG plot + prediction display
# =============================================================================
class ChildWindow(QWidget):
    closed = pyqtSignal()   # notifies parent when user closes this window

    def __init__(self, raw_data, total_windows, model, parent=None):
        super().__init__(parent)
        self.raw_data       = raw_data
        self.total_windows  = total_windows
        self.model          = model

        self.current_window  = 0
        self.elapsed_seconds = 0
        self.seg_queue       = queue.Queue()
        self.worker          = None

        # rolling buffer â€” holds last ROLLING_SAMPLES rows (averaged across display channels)
        self.roll_buf = np.zeros(ROLLING_SAMPLES, dtype=np.float32)
        self._plot_sample_ptr = 0

        # list of dicts tracking each inferred segment for annotation rolling
        self._inference_segments = []

        self.setWindowTitle("Realtime EEG Inference")
        self.setFixedSize(800, 480)
        res = QDesktopWidget().availableGeometry()
        self.move(
            (res.width()  - self.frameSize().width())  // 2,
            (res.height() - self.frameSize().height()) // 2
        )
        self.setStyleSheet(STYLESHEET)

        self._build_ui()
        self._build_timers()
        self._start()

    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # predicted word
        self.predicted_label = CustomLabel("Prediction")
        self.predicted_label.setAlignment(Qt.AlignCenter)
        self.predicted_label.setStyleSheet("font-size:50px; color:#77b9f7;")
        self.predicted_label.setFixedHeight(80)
        layout.addWidget(self.predicted_label, alignment=Qt.AlignCenter)

        # clock + inference row â€” both centre aligned
        info_row = QHBoxLayout()
        self.clip_label  = QLabel("Clip:   00:00  /  --:--")
        self.infer_label = QLabel("Inference: waiting...")
        self.clip_label.setAlignment(Qt.AlignCenter)
        self.infer_label.setAlignment(Qt.AlignCenter)
        info_row.addWidget(self.clip_label)
        info_row.addWidget(self.infer_label)
        layout.addLayout(info_row)

        # pyqtgraph rolling plot
        pg.setConfigOption('background', '#f0f8ff')
        pg.setConfigOption('foreground', '#333333')
        
        self.plot_widget = pg.GraphicsLayoutWidget()
        
        self.plot_widget.setMinimumSize(100, 100)
        self.plot_widget.setMaximumSize(16777215, 16777215)
        
        layout.addWidget(self.plot_widget)
        
        self._all_plots = []
        self._curves = []

        # --- row 0: word annotation plot ---
        self._word_plot = self.plot_widget.addPlot(row=0, col=0)
        self._word_plot.setYRange(0, 1, padding=0)
        self._word_plot.setXRange(0, ROLLING_SEC, padding=0)
        self._word_plot.getAxis('bottom').hide()
        self._word_plot.getAxis('left').setWidth(45)
        self._word_plot.getAxis('left').setStyle(showValues=False)
        self._word_plot.setLabel('left', 'Pred', color='#333333', size='9pt')
        self._word_plot.setFixedHeight(50)
        self._word_plot.setMouseEnabled(x=False, y=False)
        self._word_plot.setMenuEnabled(False)
        self._all_plots.append(self._word_plot)

        # --- row 1: single averaged EEG plot ---
        x_data = np.arange(ROLLING_SAMPLES) / SAMPLING_RATE
        p = self.plot_widget.addPlot(row=1, col=0)
        p.setYRange(Y_MIN, Y_MAX, padding=0)
        p.setXRange(0, ROLLING_SEC, padding=0)
        p.setLabel('left', 'Avg EEG', color='#333333', size='9pt')
        p.getAxis('left').setWidth(45)
        p.showGrid(x=True, y=False, alpha=0.3)
        p.setMouseEnabled(x=False, y=False)
        p.setMenuEnabled(False)
        p.setLabel('bottom', 'Time (s)')
        p.setXLink(self._word_plot)

        curve = p.plot(x_data, self.roll_buf,
                       pen=pg.mkPen(color='#2980b9', width=1.5))
        self._eeg_plot = p
        self._curves.append(curve)
        self._all_plots.append(p)

    # ------------------------------------------------------------------
    def _build_timers(self):
        # 5-sec clip timer
        self.clip_timer = QTimer()
        self.clip_timer.setInterval(WINDOW_SEC * 1000)
        self.clip_timer.timeout.connect(self._on_clip_tick)

        # 1-sec clock timer
        self.clock_timer = QTimer()
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(self._on_clock_tick)

        # plot update timer
        self.plot_timer = QTimer()
        self.plot_timer.setInterval(PLOT_INTERVAL_MS)
        self.plot_timer.timeout.connect(self._on_plot_tick)

    # ------------------------------------------------------------------
    def _start(self):
        total_sec = self.total_windows * WINDOW_SEC
        self.clip_label.setText(
            f"Clip:   {self._fmt(0)}  /  {self._fmt(total_sec)}"
        )

        # start inference worker
        self.worker = InferenceWorker(self.model, self.seg_queue)
        self.worker.result_ready.connect(self._on_inference_result)
        self.worker.start()

        # start all timers
        self.clip_timer.start()
        self.clock_timer.start()
        self.plot_timer.start()

    # ------------------------------------------------------------------
    # TIMER SLOTS
    # ------------------------------------------------------------------
    def _on_clip_tick(self):
        if self.current_window >= self.total_windows:
            self._on_clip_finished()
            return

        start_row = self.current_window * WINDOW_SAMPLES
        end_row   = start_row + WINDOW_SAMPLES
        window    = self.raw_data[start_row:end_row]

        start_sec = self.current_window * WINDOW_SEC
        end_sec   = start_sec + WINDOW_SEC

        self.seg_queue.put((self.current_window, start_sec, end_sec, window))
        print(f"[Clip] Pushed seg {self.current_window} "
              f"({self._fmt(start_sec)}â€“{self._fmt(end_sec)})")
        self.current_window += 1

        # if that was the last segment, finish immediately
        # without waiting for the next timer tick
        if self.current_window >= self.total_windows:
            self._on_clip_finished()

    def _on_clock_tick(self):
        self.elapsed_seconds += 1
        total_sec = self.total_windows * WINDOW_SEC
        self.clip_label.setText(
            f"Clip:   {self._fmt(self.elapsed_seconds)}  /  {self._fmt(total_sec)}"
        )

    def _on_plot_tick(self):
        """
        Advances _plot_sample_ptr, computes channel average,
        updates rolling buffer and curve. Annotation update always runs.
        """
        # --- data advancement ---
        total_samples = len(self.raw_data)
        if self._plot_sample_ptr < total_samples:
            end_ptr     = min(self._plot_sample_ptr + PLOT_DOWNSAMPLE, total_samples)
            new_chunk   = self.raw_data[self._plot_sample_ptr:end_ptr, :]
            # average across display channels only
            new_avg     = new_chunk[:, DISPLAY_INDICES].mean(axis=1)  # (n,)

            n = len(new_avg)
            self.roll_buf = np.roll(self.roll_buf, -n)
            self.roll_buf[-n:] = new_avg
            self._plot_sample_ptr = end_ptr

            x_data = np.arange(ROLLING_SAMPLES) / SAMPLING_RATE
            self._curves[0].setData(x_data, self.roll_buf)

        # --- annotation position update (always runs) ---
        right_edge_sample = self._plot_sample_ptr
        left_edge_sample  = right_edge_sample - ROLLING_SAMPLES

        to_remove = []
        for seg in self._inference_segments:
            x_start = (seg['start_sample'] - left_edge_sample) / SAMPLING_RATE
            x_end   = (seg['end_sample']   - left_edge_sample) / SAMPLING_RATE
            x_mid   = (x_start + x_end) / 2

            fully_off = x_end < 0 or x_start > ROLLING_SEC

            if fully_off:
                for line in seg['lines']:
                    line.hide()
                seg['text_item'].hide()
                if x_end < 0:
                    to_remove.append(seg)
            else:
                for line, x_pos in zip(seg['lines'][::2],
                                       [x_start] * len(self._all_plots)):
                    line.setPos(max(0, x_pos))
                    line.show() if x_start >= 0 else line.hide()

                for line, x_pos in zip(seg['lines'][1::2],
                                       [x_end] * len(self._all_plots)):
                    line.setPos(min(ROLLING_SEC, x_pos))
                    line.show() if x_end <= ROLLING_SEC else line.hide()

                seg['text_item'].setPos(max(0, x_mid), 0.5)
                seg['text_item'].show()

        for seg in to_remove:
            for line in seg['lines']:
                line.getViewBox().removeItem(line)
            self._word_plot.removeItem(seg['text_item'])
            self._inference_segments.remove(seg)

    # ------------------------------------------------------------------
    def _on_clip_finished(self):
        self.clip_timer.stop()
        self.clock_timer.stop()
        total_sec = self.total_windows * WINDOW_SEC
        self.clip_label.setText(
            f"Clip:   {self._fmt(total_sec)}  /  {self._fmt(total_sec)}  âœ“"
        )
        print("[Clip] Finished.")

    # ------------------------------------------------------------------
    def _on_inference_result(self, seg_idx, start_sec, end_sec, word):
        self.infer_label.setText(
            f"Inference: seg {seg_idx + 1} "
            f"({self._fmt(start_sec)}â€“{self._fmt(end_sec)}) â†’ {word}"
        )
        self._display_word(word)
        print(f"[Inference] seg {seg_idx + 1} â†’ {word}")

        # --- create annotation for this segment ---
        start_sample = start_sec * SAMPLING_RATE
        end_sample   = end_sec   * SAMPLING_RATE

        dot_pen = pg.mkPen(color='#555555', width=1.5,
                           style=Qt.DotLine)

        lines = []
        # two dotted vertical lines per plot (start + end) across all 7 plots
        for plot in self._all_plots:
            for x_pos in [start_sample, end_sample]:
                line = pg.InfiniteLine(pos=0, angle=90, pen=dot_pen, movable=False)
                plot.addItem(line)
                lines.append(line)

        # word text item on the word plot only
        text_item = pg.TextItem(
            text=word,
            color='#222222',
            anchor=(0.5, 0.5)
        )
        font = QFont("Arial", 9, QFont.Bold)
        text_item.setFont(font)
        self._word_plot.addItem(text_item)

        self._inference_segments.append({
            'start_sample': start_sample,
            'end_sample':   end_sample,
            'word':         word,
            'lines':        lines,
            'text_item':    text_item,
        })

        # if clip is done and queue drained, stop plot_timer after a short
        # delay so the final annotation gets at least one render pass
        if not self.clip_timer.isActive() and self.seg_queue.empty():
            QTimer.singleShot(3000, self.plot_timer.stop)

    # ------------------------------------------------------------------
    def _display_word(self, word):
        self.predicted_label.setText(word)
        font = QFont("Arial", 40, QFont.Bold)
        self.predicted_label.setFont(font)
        fm   = QFontMetrics(font)
        rect = fm.boundingRect(word)
        w = rect.width()  + self.predicted_label.padding * 10
        h = rect.height() + self.predicted_label.padding * 10
        self.predicted_label.setFixedSize(w, h)
        self.predicted_label.setStyleSheet("font-size:40px; color:#333;")

    # ------------------------------------------------------------------
    def _fmt(self, seconds):
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m:02d}:{s:02d}"

    # ------------------------------------------------------------------
    def _cleanup(self):
        self.clip_timer.stop()
        self.clock_timer.stop()
        self.plot_timer.stop()
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        with self.seg_queue.mutex:
            self.seg_queue.queue.clear()
        self._inference_segments.clear()

    def closeEvent(self, event):
        self._cleanup()
        self.closed.emit()   # tell parent to reset
        event.accept()


# =============================================================================
# Parent window
# =============================================================================
class ParentWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.selected_path = None
        self.model_name    = "KNN_Hindi"
        self.model         = None
        self.child_window  = None
        self.scale         = 2

        self.setObjectName("ParentWindow")
        self.setWindowTitle("Imagined Word Recognition v3")
        self.setFixedSize(800, 480)

        res = QDesktopWidget().availableGeometry()
        self.move(
            (res.width()  - self.frameSize().width())  // 2,
            (res.height() - self.frameSize().height()) // 2
        )
        self.setStyleSheet(STYLESHEET)
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):

        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.browse_button = QPushButton("Data")
        self.browse_button.setFixedSize(100 * self.scale, 38 * self.scale)
        self.browse_button.clicked.connect(self._browse)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setFixedSize(142 * self.scale, 38 * self.scale)
        self.model_combo.addItems(["KNN_Hindi"])
        le = self.model_combo.lineEdit()
        le.setAlignment(Qt.AlignCenter)
        le.setReadOnly(True)
        self.model_combo.currentTextChanged.connect(
            lambda name: setattr(self, 'model_name', name)
        )

        self.start_button = QPushButton("Start")
        self.start_button.setFixedSize(142 * self.scale, 38 * self.scale)
        self.start_button.clicked.connect(self._on_start)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedSize(142 * self.scale, 38 * self.scale)
        self.reset_button.clicked.connect(self._on_reset)

        top_row = QHBoxLayout()
        top_row.setAlignment(Qt.AlignCenter)
        top_row.addWidget(self.browse_button)

        model_row = QHBoxLayout()
        model_row.setAlignment(Qt.AlignCenter)
        model_row.addWidget(self.model_combo)

        ctrl_row = QHBoxLayout()
        ctrl_row.setAlignment(Qt.AlignCenter)
        ctrl_row.addWidget(self.start_button)
        ctrl_row.addSpacing(20)
        ctrl_row.addWidget(self.reset_button)

        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addLayout(top_row)
        layout.addSpacing(10)
        layout.addLayout(model_row)
        layout.addSpacing(10)
        layout.addLayout(ctrl_row)
        layout.addSpacing(15)
        layout.addWidget(self.status_label, alignment=Qt.AlignCenter)
        layout.addStretch()

    # ------------------------------------------------------------------
    def _update_status(self, text):
        self.status_label.setText(f"Status: {text}")
        QApplication.processEvents()

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.selected_path = folder
            self._update_status(f"Loaded: {Path(folder).name}")

    # ------------------------------------------------------------------
    def _on_start(self):
        if not self.selected_path:
            self._update_status("No data folder selected")
            return

        # load data
        try:
            self._update_status("Loading data...")
            raw_data, total_windows = self._load_data()
        except Exception as e:
            self._update_status(f"Data error: {e}")
            return

        if total_windows == 0:
            self._update_status("File too short (< 5 sec)")
            return

        # load model
        try:
            self._update_status("Loading model...")
            self.model = KNN_Hindi_model()
            self.model.load_model()
        except Exception as e:
            self._update_status(f"Model error: {e}")
            return

        # hide parent, open child
        self.hide()
        self.child_window = ChildWindow(raw_data, total_windows, self.model)
        self.child_window.closed.connect(self._on_child_closed)
        self.child_window.show()

    def _load_data(self):
        data_path = Path(self.selected_path) / "data.csv"
        df   = pd.read_csv(data_path)
        data = df.iloc[:, 1:].astype(np.float32).values
        data = np.nan_to_num(data)
        mean = np.mean(data, axis=0, keepdims=True)
        std  = np.std(data,  axis=0, keepdims=True)
        std[std == 0] = 1
        data = (data - mean) / std
        total_windows = len(data) // WINDOW_SAMPLES
        print(f"Total rows: {len(data)}, windows: {total_windows}")
        return data, total_windows

    # ------------------------------------------------------------------
    def _on_child_closed(self):
        """Child was closed â€” reset parent fully and show it again."""
        self.child_window = None
        self._on_reset()
        self.show()

    def _on_reset(self):
        self.selected_path = None
        self.model         = None
        self._update_status("Idle")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self.child_window is not None:
            self.child_window._cleanup()
            self.child_window.close()
        event.accept()


# =============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ParentWindow()
    win.show()
    sys.exit(app.exec_())

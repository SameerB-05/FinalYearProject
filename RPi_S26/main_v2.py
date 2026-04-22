import sys
import queue
import threading
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QComboBox, QVBoxLayout, QHBoxLayout, QLabel,
    QStackedWidget, QDesktopWidget
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QFontMetrics, QPainter, QPainterPath, QPen, QColor, QIcon
from PyQt5.QtCore import QRectF

import numpy as np
import pandas as pd

from models.knn_hindi.knn_hindi import KNN_Hindi_model


# =============================================================================
# Custom rounded-border label (same as main.py)
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
            self.padding // 2,
            self.padding // 2,
            -self.padding // 2,
            -self.padding // 2
        )
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 20, 20)
        painter.setPen(QPen(QColor("#000000"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        super().paintEvent(event)


# =============================================================================
# Inference worker — runs in background thread, drains the segment queue
# =============================================================================
class InferenceWorker(QThread):
    # emits (segment_index, start_sec, end_sec, predicted_word)
    result_ready = pyqtSignal(int, int, int, str)

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
                # window shape: (1280, 32) → wrap in batch dim → (1, 1280, 32)
                batch  = window[np.newaxis, :, :]
                epochs = self.model._preprocess(batch)     # (n_epochs, 256, 32)
                word   = self.model.predict(epochs)
                word   = "PAIN" if word == 0 else "LIGHT"
            except Exception as e:
                word = f"ERR: {e}"

            self.result_ready.emit(seg_idx, start_sec, end_sec, word)
            self.seg_queue.task_done()

    def stop(self):
        self._running = False


# =============================================================================
# Main window v2
# =============================================================================
class MainWindowV2(QWidget):
    def __init__(self):
        super().__init__()

        self.selected_path  = None
        self.model_name     = "KNN_Hindi"
        self.model          = None
        self.scale          = 2

        # clip state
        self.raw_data        = None   # full (N, 32) numpy array
        self.total_windows   = 0
        self.current_window  = 0      # clip pointer
        self.elapsed_seconds = 0      # seconds elapsed since start (for clock display)
        self.seg_queue       = queue.Queue()
        self.worker          = None

        # 5-sec timer — advances clip pointer, pushes segment to queue
        self.clip_timer = QTimer()
        self.clip_timer.setInterval(5000)
        self.clip_timer.timeout.connect(self._on_clip_tick)

        # 1-sec timer — updates the digital clock display only
        self.clock_timer = QTimer()
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(self._on_clock_tick)

        self.setObjectName("MainWindowV2")
        self.setWindowTitle("Imagined Word Recognition — Realtime v2")
        self.setFixedSize(600, 500)

        res = QDesktopWidget().availableGeometry()
        self.move(
            (res.width()  - self.frameSize().width())  // 2,
            (res.height() - self.frameSize().height()) // 2
        )

        self.setStyleSheet("""
            QWidget#MainWindowV2 {
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
                width: 12px;
                height: 12px;
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
        """)

        self._build_ui()

    # =========================================================
    # UI BUILD
    # =========================================================
    def _build_ui(self):

        # --- predicted word label ---
        self.predicted_label = CustomLabel("Prediction")
        self.predicted_label.setAlignment(Qt.AlignCenter)
        self.predicted_label.setStyleSheet("font-size:50px; color:#77b9f7;")

        # --- clip progress label ---
        self.clip_label = QLabel("Clip:       --:-- / --:--")
        self.clip_label.setAlignment(Qt.AlignCenter)

        # --- inference progress label ---
        self.infer_label = QLabel("Inference: waiting...")
        self.infer_label.setAlignment(Qt.AlignCenter)

        # --- status label ---
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)

        # --- browse button ---
        self.browse_button = QPushButton("Data")
        self.browse_button.setFixedSize(100 * self.scale, 38 * self.scale)
        self.browse_button.clicked.connect(self.browse_folder)

        # --- model combo ---
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setFixedSize(142 * self.scale, 38 * self.scale)
        self.model_combo.addItems(["KNN_Hindi"])
        le = self.model_combo.lineEdit()
        le.setAlignment(Qt.AlignCenter)
        le.setReadOnly(True)
        self.model_combo.currentTextChanged.connect(self._select_model)

        # --- control buttons ---
        self.start_button  = QPushButton("Start")
        self.start_button.setFixedSize(90 * self.scale, 38 * self.scale)
        self.start_button.clicked.connect(self._on_start)

        self.pause_button  = QPushButton("Pause")
        self.pause_button.setFixedSize(90 * self.scale, 38 * self.scale)
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._on_pause)

        self.reset_button  = QPushButton("Reset")
        self.reset_button.setFixedSize(90 * self.scale, 38 * self.scale)
        self.reset_button.clicked.connect(self._on_reset)

        # --- layouts ---
        top_row = QHBoxLayout()
        top_row.setAlignment(Qt.AlignCenter)
        top_row.addWidget(self.browse_button)

        model_row = QHBoxLayout()
        model_row.setAlignment(Qt.AlignCenter)
        model_row.addWidget(self.model_combo)

        ctrl_row = QHBoxLayout()
        ctrl_row.setAlignment(Qt.AlignCenter)
        ctrl_row.addWidget(self.start_button)
        ctrl_row.addSpacing(10)
        ctrl_row.addWidget(self.pause_button)
        ctrl_row.addSpacing(10)
        ctrl_row.addWidget(self.reset_button)

        main_layout = QVBoxLayout(self)
        main_layout.addStretch()
        main_layout.addWidget(self.predicted_label, alignment=Qt.AlignCenter)
        main_layout.addSpacing(15)
        main_layout.addWidget(self.clip_label)
        main_layout.addWidget(self.infer_label)
        main_layout.addSpacing(10)
        main_layout.addLayout(top_row)
        main_layout.addSpacing(5)
        main_layout.addLayout(model_row)
        main_layout.addSpacing(10)
        main_layout.addLayout(ctrl_row)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.status_label, alignment=Qt.AlignCenter)
        main_layout.addStretch()

    # =========================================================
    # HELPERS
    # =========================================================
    def _fmt_sec(self, seconds):
        m = seconds // 60
        s = seconds % 60
        return f"{m:02d}:{s:02d}"

    def _on_clock_tick(self):
        """Fires every 1 second — updates the digital clock display."""
        self.elapsed_seconds += 1
        total_sec = self.total_windows * 5
        self.clip_label.setText(
            f"Clip:   {self._fmt_sec(self.elapsed_seconds)}  /  {self._fmt_sec(total_sec)}"
        )

    def _update_status(self, text):
        self.status_label.setText(f"Status: {text}")
        QApplication.processEvents()

    def _select_model(self, name):
        self.model_name = name

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, 'predicted_label'):
            return
        text = self.predicted_label.text()
        if text and text != "Prediction":
            font = QFont("Arial", 40, QFont.Bold)
            self.predicted_label.setFont(font)
            fm   = QFontMetrics(font)
            rect = fm.boundingRect(text)
            w = rect.width()  + self.predicted_label.padding * 10
            h = rect.height() + self.predicted_label.padding * 10
            self.predicted_label.setFixedSize(w, h)
            self.predicted_label.setStyleSheet("font-size:40px; color:#333;")

    # =========================================================
    # BROWSE
    # =========================================================
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.selected_path = folder
            self._update_status(f"Loaded: {Path(folder).name}")
            print(f"Selected path: {folder}")

    # =========================================================
    # LOAD DATA + MODEL
    # =========================================================
    def _load_data(self):
        data_path = Path(self.selected_path) / "data.csv"
        df   = pd.read_csv(data_path)
        data = df.iloc[:, 1:].astype(np.float32).values   # drop timestamp col
        data = np.nan_to_num(data)

        # z-score normalize per channel (same as knn_hindi predict_from_folder)
        mean = np.mean(data, axis=0, keepdims=True)
        std  = np.std(data,  axis=0, keepdims=True)
        std[std == 0] = 1
        data = (data - mean) / std

        self.raw_data      = data
        self.total_windows = len(data) // 1280
        print(f"Total rows: {len(data)}, Total 5-sec windows: {self.total_windows}")

    def _load_model(self):
        self.model = KNN_Hindi_model()
        self.model.load_model()

    # =========================================================
    # CLIP TIMER TICK
    # =========================================================
    def _on_clip_tick(self):
        if self.current_window >= self.total_windows:
            self._on_clip_finished()
            return

        start_row = self.current_window * 1280
        end_row   = start_row + 1280
        window    = self.raw_data[start_row:end_row]   # (1280, 32)

        start_sec = self.current_window * 5
        end_sec   = start_sec + 5

        # push to inference queue
        self.seg_queue.put((self.current_window, start_sec, end_sec, window))
        print(f"[Clip] Pushed segment {self.current_window} "
              f"({self._fmt_sec(start_sec)}–{self._fmt_sec(end_sec)})")

        self.current_window += 1

    def _on_clip_finished(self):
        self.clip_timer.stop()
        self.clock_timer.stop()
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        total_sec = self.total_windows * 5
        self.clip_label.setText(
            f"Clip:   {self._fmt_sec(total_sec)}  /  {self._fmt_sec(total_sec)}  ✓"
        )
        self._update_status("Clip finished — inference draining queue...")

    # =========================================================
    # INFERENCE RESULT SLOT
    # =========================================================
    def _on_inference_result(self, seg_idx, start_sec, end_sec, word):
        self.infer_label.setText(
            f"Inference: seg {seg_idx}  "
            f"({self._fmt_sec(start_sec)}–{self._fmt_sec(end_sec)})  →  {word}"
        )
        self._display_word(word)
        print(f"[Inference] Segment {seg_idx} "
              f"({self._fmt_sec(start_sec)}–{self._fmt_sec(end_sec)}) → {word}")

        # if clip is done and queue is now empty, fully complete
        if not self.clip_timer.isActive() and self.seg_queue.empty():
            self._update_status("Completed")

    # =========================================================
    # CONTROLS
    # =========================================================
    def _on_start(self):
        if not self.selected_path:
            self._update_status("No data folder selected")
            return

        # fresh start — reset state
        self._stop_worker()
        self.current_window = 0
        with self.seg_queue.mutex:
            self.seg_queue.queue.clear()

        try:
            self._update_status("Loading data...")
            self._load_data()
            self._update_status("Loading model...")
            self._load_model()
        except Exception as e:
            self._update_status(f"Error: {e}")
            return

        if self.total_windows == 0:
            self._update_status("File too short (< 5 sec)")
            return

        # start inference worker
        self.worker = InferenceWorker(self.model, self.seg_queue)
        self.worker.result_ready.connect(self._on_inference_result)
        self.worker.start()

        # reset clock and start both timers
        self.elapsed_seconds = 0
        total_sec = self.total_windows * 5
        self.clip_label.setText(
            f"Clip:   {self._fmt_sec(0)}  /  {self._fmt_sec(total_sec)}"
        )
        self.clip_timer.start()
        self.clock_timer.start()
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.pause_button.setText("Pause")
        self._update_status("Running...")

    def _on_pause(self):
        if self.clip_timer.isActive():
            self.clip_timer.stop()
            self.clock_timer.stop()
            self.pause_button.setText("Resume")
            self._update_status("Paused")
        else:
            self.clip_timer.start()
            self.clock_timer.start()
            self.pause_button.setText("Pause")
            self._update_status("Running...")

    def _on_reset(self):
        # stop everything
        self.clip_timer.stop()
        self.clock_timer.stop()
        self._stop_worker()

        # clear queue
        with self.seg_queue.mutex:
            self.seg_queue.queue.clear()

        # reset state
        self.current_window  = 0
        self.elapsed_seconds = 0
        self.raw_data        = None
        self.total_windows   = 0

        # reset UI
        self.predicted_label.setText("Prediction")
        self.predicted_label.setFixedSize(200 * self.scale, 38 * self.scale)
        self.predicted_label.setStyleSheet("font-size:50px; color:#77b9f7;")
        self.clip_label.setText("Clip:   --:--  /  --:--")
        self.infer_label.setText("Inference: waiting...")
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self._update_status("Idle")

    def _stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)   # wait up to 3sec for thread to exit
            self.worker = None

    # =========================================================
    # CLOSE EVENT — clean up thread on window close
    # =========================================================
    def closeEvent(self, event):
        self.clip_timer.stop()
        self.clock_timer.stop()
        self._stop_worker()
        event.accept()


# =============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindowV2()
    win.show()
    sys.exit(app.exec_())
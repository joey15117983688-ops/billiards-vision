"""Play a local billiards-table video in the existing PyQt5 window."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import sys

import numpy as np
import cv2
from PyQt5.QtCore import QObject, QRectF, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)
from src.yolo_labels import chinese_label

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "data" / "billiards_table.mp4"


class YoloWorker(QObject):
    ready = pyqtSignal()
    result = pyqtSignal(int, str, float, object)
    failed = pyqtSignal(int, str)

    def __init__(self):
        super().__init__()
        self.process = None
        self.log = None

    def read_reply(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("YOLO 推理进程已退出，请重新开启 YOLO；详情见 logs/yolo_ui.log")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    @pyqtSlot()
    def load(self):
        try:
            self.shutdown()
            logs = ROOT / "logs"
            logs.mkdir(exist_ok=True)
            self.log = (logs / "yolo_ui.log").open("a")
            # Run tensor operations in a separate process's main thread on macOS.
            self.process = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / "yolo_click_worker.py")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, bufsize=1, cwd=str(ROOT),
            )
            if not self.read_reply().get("ready"):
                raise RuntimeError("YOLO 模型未就绪")
            self.ready.emit()
        except Exception as exc:
            self.failed.emit(-1, str(exc))

    @pyqtSlot(object, float, float, int)
    def identify(self, frame, x, y, generation):
        path = None
        try:
            with tempfile.NamedTemporaryFile(dir=ROOT / "logs", suffix=".npy", delete=False) as f:
                np.save(f, frame, allow_pickle=False)
                path = Path(f.name)
            self.process.stdin.write(json.dumps({"frame": str(path), "x": x, "y": y}) + "\n")
            self.process.stdin.flush()
            reply = self.read_reply()
            self.result.emit(generation, reply["name"], reply["confidence"], reply.get("box"))
        except Exception as exc:
            self.failed.emit(generation, str(exc))
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def shutdown(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None


class VideoCanvas(QWidget):
    element_clicked = pyqtSignal(object, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.frame = None
        self.badge_text = "1"
        self.selection_enabled = False
        self.selected_box = None
        self.selected_label = ""
        self.message = "正在打开视频…"
        self.setMinimumSize(320, 240)
        self.setAccessibleName("本地演示视频，右上角白底圆圈数字1")

    def set_frame(self, frame):
        self.clear_selection()
        self.frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        self.image = QImage(rgb.data, width, height, rgb.strides[0],
                            QImage.Format_RGB888).copy()
        self.update()

    def clear_selection(self):
        self.selected_box = None
        self.selected_label = ""
        self.update()

    def set_selection(self, box, label):
        self.selected_box = box
        self.selected_label = label
        self.update()

    def selected_rect(self):
        if self.selected_box is None or self.image.isNull():
            return QRectF()
        x1, y1, x2, y2 = self.selected_box
        target = self.video_rect()
        sx, sy = target.width() / self.image.width(), target.height() / self.image.height()
        return QRectF(target.left() + x1 * sx, target.top() + y1 * sy,
                      (x2 - x1) * sx, (y2 - y1) * sy)

    def video_rect(self):
        size = self.image.size().scaled(self.size(), Qt.KeepAspectRatio)
        return QRectF((self.width() - size.width()) / 2,
                      (self.height() - size.height()) / 2,
                      size.width(), size.height())

    def badge_rect(self):
        return QRectF(self.width() - 176, 16, 160, 160)

    def set_badge(self, text):
        self.badge_text = text
        self.setAccessibleName(f"视频右上角识别结果：{text}")
        self.update()

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton and self.selection_enabled
                and self.frame is not None and not self.image.isNull()):
            point = event.localPos()
            target = self.video_rect()
            if target.contains(point) and not self.badge_rect().contains(point):
                x = (point.x() - target.left()) * self.image.width() / target.width()
                y = (point.y() - target.top()) * self.image.height() / target.height()
                self.element_clicked.emit(self.frame.copy(), x, y)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.image.isNull():
            painter.setPen(Qt.white)
            painter.drawText(self.rect().adjusted(24, 24, -24, -24),
                             Qt.AlignCenter | Qt.TextWordWrap, self.message)
        else:
            painter.drawImage(self.video_rect(), self.image)
            if self.selected_box is not None:
                rect = self.selected_rect()
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(Qt.black, 8))
                painter.drawRect(rect)
                highlight = QColor("#FFEB00")
                painter.setPen(QPen(highlight, 4))
                painter.drawRect(rect)
                label_font = QFont(self.font())
                label_font.setPixelSize(20)
                label_font.setBold(True)
                painter.setFont(label_font)
                label_width = painter.fontMetrics().horizontalAdvance(self.selected_label) + 20
                label = QRectF(
                    min(max(0, rect.left()), self.width() - label_width),
                    max(self.video_rect().top(), rect.top() - 32), label_width, 32)
                painter.fillRect(label, Qt.black)
                painter.setPen(highlight)
                painter.drawText(label, Qt.AlignCenter, self.selected_label)
        # An opaque white badge remains readable over every frame.
        painter.setRenderHint(QPainter.Antialiasing)
        circle = self.badge_rect()
        painter.setPen(QPen(Qt.black, 3))
        painter.setBrush(Qt.white)
        painter.drawEllipse(circle)
        font = QFont(self.font())
        font.setPixelSize(64 if self.badge_text == "1" else 24)
        painter.setFont(font)
        text_rect = circle.adjusted(20, 28, -20, -28)
        while font.pixelSize() > 14:
            bounds = painter.fontMetrics().boundingRect(
                text_rect.toRect(), Qt.AlignCenter | Qt.TextWordWrap, self.badge_text)
            if bounds.width() <= text_rect.width() and bounds.height() <= text_rect.height():
                break
            font.setPixelSize(font.pixelSize() - 1)
            painter.setFont(font)
        painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, self.badge_text)


class MainWindow(QMainWindow):
    identify_requested = pyqtSignal(object, float, float, int)
    load_requested = pyqtSignal()

    def __init__(self, video_path=DEFAULT_VIDEO):
        super().__init__()
        self.setWindowTitle("台球视觉识别系统")
        self.resize(1200, 800)
        self.canvas = VideoCanvas()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        controls = QWidget()
        buttons = QHBoxLayout(controls)
        buttons.setContentsMargins(16, 8, 16, 8)
        buttons.setSpacing(12)
        self.play_button = QPushButton("播放")
        self.pause_button = QPushButton("暂停")
        self.yolo_button = QPushButton("YOLO：关闭")
        self.yolo_button.setCheckable(True)
        self.yolo_button.setMinimumSize(150, 44)
        self.yolo_button.setToolTip("开启台球专用模型后，点击球桌或球；首次点击会暂停画面")
        buttons.addStretch()
        for button in (self.play_button, self.pause_button):
            button.setMinimumSize(100, 44)
            button.setEnabled(False)
            buttons.addWidget(button)
        buttons.addWidget(self.yolo_button)
        buttons.addStretch()
        layout.addWidget(controls)
        self.setCentralWidget(content)
        self.capture = None
        self.playback_ready = False
        self.yolo_thread = None
        self.yolo_worker = None
        self.yolo_ready = False
        self.yolo_loading = False
        self.yolo_busy = False
        self.yolo_generation = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.yolo_button.toggled.connect(self.toggle_yolo)
        self.canvas.element_clicked.connect(self.identify_element)
        self.video_path = Path(video_path).expanduser().resolve()
        if not self.video_path.is_file():
            self.show_error(f"未找到视频：{self.video_path}\n请先运行 scripts/download_demo_assets.py。")
            return
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            self.show_error(f"无法打开视频：{self.video_path.name}")
            self.capture.release()
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.next_frame()  # Display a decoded frame immediately, before the first timer tick.
        if not self.canvas.image.isNull():
            self.playback_ready = True
            self.timer.setInterval(round(1000 / fps) if 1 <= fps <= 120 else 40)
            self.play_video()

    def toggle_yolo(self, enabled):
        self.yolo_generation += 1
        self.canvas.selection_enabled = enabled
        self.canvas.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.yolo_button.setText("YOLO：开启" if enabled else "YOLO：关闭")
        if not enabled:
            self.canvas.clear_selection()
            self.canvas.set_badge("1")
            self.statusBar().showMessage("YOLO 已关闭")
            return
        if self.yolo_ready:
            self.canvas.set_badge("识别中…" if self.yolo_busy else "请点击\n物体")
            self.statusBar().showMessage("正在完成上一次推理…" if self.yolo_busy
                                         else "YOLO 已开启，点击视频中的物体；可暂停后选择")
            return
        self.canvas.set_badge("加载中…")
        self.statusBar().showMessage("正在加载 YOLO，视频可以继续播放")
        if self.yolo_thread is None:
            self.yolo_thread = QThread(self)
            self.yolo_worker = YoloWorker()
            self.yolo_worker.moveToThread(self.yolo_thread)
            self.yolo_thread.finished.connect(self.yolo_worker.deleteLater)
            self.identify_requested.connect(self.yolo_worker.identify)
            self.load_requested.connect(self.yolo_worker.load)
            self.yolo_worker.ready.connect(self.on_yolo_ready)
            self.yolo_worker.result.connect(self.on_yolo_result)
            self.yolo_worker.failed.connect(self.on_yolo_error)
            self.yolo_thread.start()
        if not self.yolo_loading:
            self.yolo_loading = True
            self.load_requested.emit()

    def on_yolo_ready(self):
        self.yolo_ready = True
        self.yolo_loading = False
        if self.yolo_button.isChecked():
            self.canvas.set_badge("请点击\n物体")
            self.statusBar().showMessage("YOLO 已开启，点击视频中的物体；可暂停后选择")

    def identify_element(self, frame, x, y):
        if not self.yolo_button.isChecked() or not self.yolo_ready or self.yolo_busy:
            return
        self.yolo_busy = True
        self.pause_video()
        self.canvas.set_frame(frame)
        self.canvas.set_badge("识别中…")
        self.statusBar().showMessage("正在识别点击时的画面…")
        self.identify_requested.emit(frame, x, y, self.yolo_generation)

    def on_yolo_result(self, generation, name, confidence, box=None):
        self.yolo_busy = False
        if not self.yolo_button.isChecked():
            return
        if generation != self.yolo_generation:
            self.canvas.set_badge("请点击\n物体")
            self.statusBar().showMessage("YOLO 已开启，点击视频中的物体；可暂停后选择")
            return
        if name:
            label = chinese_label(name)
            self.canvas.set_badge(label)
            self.canvas.set_selection(box, f"{label} {confidence:.0%}")
            self.statusBar().showMessage(f"点击位置识别为：{label} · 置信度 {confidence:.0%} · 已暂停，可点击播放继续")
        else:
            self.canvas.set_badge("未识别")
            self.statusBar().showMessage("台球专用模型未检测到点击位置的物体，可尝试更清晰的画面")

    def on_yolo_error(self, generation, message):
        self.yolo_busy = False
        if generation == -1:
            self.yolo_loading = False
            self.yolo_ready = False
        elif self.yolo_worker.process is None or self.yolo_worker.process.poll() is not None:
            self.yolo_ready = False
        if self.yolo_button.isChecked() and (generation == -1 or generation == self.yolo_generation):
            self.canvas.set_badge("识别失败")
            self.statusBar().showMessage(f"YOLO 错误：{message}")
        elif self.yolo_button.isChecked():
            self.canvas.set_badge("请点击\n物体")
            self.statusBar().showMessage("YOLO 已开启，点击视频中的物体；可暂停后选择")

    def play_video(self):
        if not self.playback_ready:
            return
        self.yolo_generation += 1
        self.canvas.clear_selection()
        if self.yolo_button.isChecked() and self.yolo_ready:
            self.canvas.set_badge("识别中…" if self.yolo_busy else "请点击\n物体")
        self.timer.start()
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.statusBar().showMessage(f"本地视频：{self.video_path.name} · 正在循环播放")

    def pause_video(self):
        if not self.playback_ready:
            return
        self.timer.stop()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.statusBar().showMessage(f"本地视频：{self.video_path.name} · 已暂停")

    def show_error(self, message):
        self.timer.stop()
        self.playback_ready = False
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.canvas.message = message
        self.canvas.update()
        self.statusBar().showMessage("视频播放失败")
        print(message, file=sys.stderr, flush=True)

    def next_frame(self):
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if not ok:
            self.canvas.image = QImage()
            self.show_error("无法读取视频帧，请检查视频文件是否完整。")
            return
        self.canvas.set_frame(frame)

    def closeEvent(self, event):
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
        if self.yolo_thread is not None:
            if self.yolo_worker.process is not None:
                self.yolo_worker.process.terminate()
            self.yolo_thread.quit()
            self.yolo_thread.wait()
            self.yolo_worker.shutdown()
        super().closeEvent(event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="播放本地台球球桌测试视频")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    args = parser.parse_args()
    app = QApplication([sys.argv[0]])
    window = MainWindow(args.video)
    window.show()
    sys.exit(app.exec_())

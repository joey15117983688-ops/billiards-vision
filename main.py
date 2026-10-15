"""Play a local snooker match recording in the existing PyQt5 window."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import time
from collections import deque
import subprocess
import tempfile
from pathlib import Path
import sys

import numpy as np
import cv2
from PyQt5.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPolygonF, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QMainWindow, QPushButton, QShortcut, QVBoxLayout, QWidget,
)
from src.yolo_labels import chinese_label
from src.table_coordinates import TableCoordinates, TableDimensions
from src.live_tracking import LiveTracking
from src.track_recording import TrackRecorder

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "data" / "overhead_pool.mp4"
DEFAULT_IMAGES = ROOT / "data" / "dataset" / "images" / "train"


class YoloWorker(QObject):
    ready = pyqtSignal()
    result = pyqtSignal(int, str, float, object, object)
    all_result = pyqtSignal(int, object)
    live_result = pyqtSignal(int, object)
    live_failed = pyqtSignal(int, str)
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

    def request_prediction(self, frame, options):
        path = None
        try:
            with tempfile.NamedTemporaryFile(dir=ROOT / "logs", suffix=".npy", delete=False) as f:
                np.save(f, frame, allow_pickle=False)
                path = Path(f.name)
            self.process.stdin.write(json.dumps({"frame": str(path), **options}) + "\n")
            self.process.stdin.flush()
            return self.read_reply()
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    @pyqtSlot(object, float, float, int)
    def identify(self, frame, x, y, generation):
        try:
            reply = self.request_prediction(frame, {"x": x, "y": y})
            self.result.emit(generation, reply["name"], reply["confidence"], reply.get("box"), reply.get("polygon"))
        except Exception as exc:
            self.failed.emit(generation, str(exc))

    @pyqtSlot(object, int)
    def identify_all(self, frame, generation):
        try:
            reply = self.request_prediction(frame, {"mode": "all"})
            self.all_result.emit(generation, reply["detections"])
        except Exception as exc:
            self.failed.emit(generation, str(exc))

    @pyqtSlot(object, int)
    def identify_live(self, frame, token):
        try:
            options={"mode":"all"}
            if isinstance(frame,dict):
                options={"mode":"selected","targets":frame["targets"]} if "targets" in frame else {"mode":"single","target":frame["target"]}
                frame=frame["frame"]
            reply=self.request_prediction(frame,options)
            self.live_result.emit(token,reply["detections"])
        except Exception as exc:
            self.live_failed.emit(token,str(exc))

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


class PocketWorker(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    @pyqtSlot(object, int, object)
    def detect(self, frame, generation, source):
        try:
            from src.vision.pocket import PocketDetector
            from src.vision.pocket.debug import save_debug_result
            if QThread.currentThread().isInterruptionRequested():
                return
            result = PocketDetector().detect(frame)
            result.source = dict(source)
            if QThread.currentThread().isInterruptionRequested():
                return
            stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
            folder = ROOT / 'data' / 'debug' / 'pocket_detection' / stamp
            save_debug_result(frame, result, folder)
            self.done.emit(generation, result.to_dict())
        except Exception as exc:
            self.failed.emit(generation, str(exc))


class VideoCanvas(QWidget):
    element_clicked = pyqtSignal(object, float, float)
    coordinate_hovered = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.frame = None
        self.coordinate_mode = False
        self.coordinates = None
        self.dimensions = TableDimensions()
        self.coordinate_ball = None
        self.setMouseTracking(True)
        self.selection_enabled = False
        self.selected_box = None
        self.selected_polygon = None
        self.selected_label = ""
        self.detections = []
        self.pocket_result = None
        self.track_paths={}
        self.show_trails=True
        self.marked_ids=set()
        self.message = "正在打开视频…"
        self.setMinimumSize(320, 240)
        self.setAccessibleName("台球视频或样本图片，可开启 YOLO 后点击识别物体")

    def set_frame(self, frame, keep_pockets=False):
        if not keep_pockets:
            self.pocket_result = None
        self.clear_selection()
        self.frame = frame
        self.coordinates = None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        self.image = QImage(rgb.data, width, height, rgb.strides[0],
                            QImage.Format_RGB888).copy()
        if self.coordinate_mode:
            self.calibrate_coordinates()
        self.update()

    def set_pocket_result(self, result):
        self.pocket_result = result
        if self.coordinate_mode:
            self.calibrate_coordinates()
        self.update()

    def clear_pockets(self):
        self.pocket_result = None
        self.update()

    def clear_selection(self):
        self.coordinate_ball = None
        self.selected_box = None
        self.selected_polygon = None
        self.selected_label = ""
        self.detections = []
        self.update()

    def set_selection(self, box, label, polygon=None):
        self.detections = []
        self.selected_box = box
        self.selected_polygon = polygon
        self.selected_label = label
        self.update()

    def set_detections(self, detections):
        self.clear_selection()
        self.detections = detections
        self.update()

    def selected_rect(self):
        return self.box_rect(self.selected_box)

    def box_rect(self, box):
        if box is None or self.image.isNull():
            return QRectF()
        x1, y1, x2, y2 = box
        target = self.video_rect()
        sx, sy = target.width() / self.image.width(), target.height() / self.image.height()
        return QRectF(target.left() + x1 * sx, target.top() + y1 * sy,
                      (x2 - x1) * sx, (y2 - y1) * sy)

    def video_rect(self):
        size = self.image.size().scaled(self.size(), Qt.KeepAspectRatio)
        return QRectF((self.width() - size.width()) / 2,
                      (self.height() - size.height()) / 2,
                      size.width(), size.height())

    def calibrate_coordinates(self):
        self.coordinates = None
        self.coordinate_ball = None
        if self.frame is None:
            self.coordinate_hovered.emit("尚无画面")
            return
        from src.target_balls import playing_surface
        polygon = None
        if self.pocket_result and self.pocket_result['table']['quality'].get('geometry_reliable'):
            polygon = self.pocket_result['table']['playable_polygon']
        if polygon is None:
            polygon = playing_surface(self.frame)
        try:
            if polygon is not None:
                self.coordinates = TableCoordinates(polygon,self.dimensions)
        except (ValueError,cv2.error):
            self.coordinates = None
        if self.coordinate_mode:
            self.coordinate_hovered.emit("移动鼠标查看台面毫米坐标" if self.coordinates else "未可靠定位内库，暂不可换算坐标")

    def mouseMoveEvent(self, event):
        if self.coordinate_mode:
            area=self.video_rect();point=event.localPos()
            position=None
            if area.contains(point) and self.coordinates is not None:
                x=(point.x()-area.left())*self.image.width()/area.width()
                y=(point.y()-area.top())*self.image.height()/area.height()
                position=self.coordinates.millimetres(x,y)
            text=(f"鼠标：X = {position[0]:.1f} mm，Y = {position[1]:.1f} mm" if position
                  else "鼠标在台面外" if self.coordinates else "未可靠定位内库，暂不可换算坐标")
            self.coordinate_hovered.emit(text)
        super().mouseMoveEvent(event)

    def leaveEvent(self,event):
        if self.coordinate_mode:self.coordinate_hovered.emit("鼠标已离开画面")
        super().leaveEvent(event)

    def draw_coordinates(self,painter):
        if not self.coordinate_mode or self.coordinates is None:return
        d=self.dimensions
        def mapped(points):return self.canvas_polygon(self.coordinates.pixels(points))
        origin_x=mapped([[0,0],[d.x_mm,0]])
        origin_y=mapped([[0,0],[0,d.y_mm]])
        painter.setPen(QPen(QColor('#F6C85F'),3));painter.setBrush(Qt.NoBrush)
        painter.drawLine(origin_x[0],origin_x[1]);painter.drawLine(origin_y[0],origin_y[1])
        font=QFont(self.font());font.setPixelSize(14);font.setBold(True);painter.setFont(font)
        labels=[('O (0, 0) mm',origin_x[0]),('+X',origin_x[1]),('+Y',origin_y[1])]
        area=self.video_rect()
        for text,point in labels:
            width=painter.fontMetrics().horizontalAdvance(text)+12
            rect=QRectF(min(max(point.x()+8,area.left()),max(area.left(),area.right()-width)),
                        min(max(point.y()+8,area.top()),area.bottom()-26),width,26)
            painter.fillRect(rect,Qt.black);painter.drawText(rect,Qt.AlignCenter,text)
        if self.coordinate_ball:
            x,y=self.coordinate_ball
            lines=mapped([[x,0],[x,y],[0,y]])
            painter.setPen(QPen(QColor('#F6C85F'),2,Qt.DashLine))
            painter.drawLine(lines[0],lines[1]);painter.drawLine(lines[1],lines[2])
            painter.drawEllipse(lines[1],6,6)

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton and self.selection_enabled
                and self.frame is not None and not self.image.isNull()):
            point = event.localPos()
            target = self.video_rect()
            if target.contains(point):
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
            self.draw_track_paths(painter)
            occupied = []
            # Draw large table boxes first so small ball boxes stay visible.
            items = sorted(self.detections, key=lambda d: (d["box"][2]-d["box"][0]) *
                           (d["box"][3]-d["box"][1]), reverse=True)
            colors = {"billiards_table": "#40C4FF", "white_ball": "#B2FF59", "black_ball": "#FF9F43"}
            for item in items:
                if self.selected_box is not None and item["box"]==self.selected_box:
                    continue
                self.draw_detection(painter, item["box"], chinese_label(item["name"])+(f" B{item['ball_id']}" if item.get("ball_id") else "")+(" · 已选" if item.get("track_id") in self.marked_ids else ""),
                                    ("#FF9F43" if item.get("track_id") in self.marked_ids else colors.get(item["name"], "#FFEB00")), 14, occupied, item.get("polygon"))
            if self.selected_box is not None:
                self.draw_detection(painter, self.selected_box, self.selected_label,
                                    "#FFEB00", 20, occupied, self.selected_polygon)
            self.draw_pockets(painter)
            self.draw_coordinates(painter)

    def draw_track_paths(self,painter):
        if not self.show_trails:return
        painter.setRenderHint(QPainter.Antialiasing);painter.setBrush(Qt.NoBrush)
        for key,points in self.track_paths.items():
            painter.setPen(QPen(QColor.fromHsv((int(key)*137)%360,210,245),2))
            segment=[]
            for x,y,gap in points:
                if gap and segment:
                    if len(segment)>1:painter.drawPolyline(self.canvas_polygon(segment))
                    segment=[]
                segment.append([x,y])
            if len(segment)>1:painter.drawPolyline(self.canvas_polygon(segment))

    def draw_pockets(self, painter):
        result = self.pocket_result
        if not result:
            return
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        table = result['table']
        scope=table.get('scope_polygon') or table['playable_polygon']
        if scope:
            painter.setPen(QPen(QColor('#40C4FF'), 2))
            painter.drawPolygon(self.canvas_polygon(scope))
            for edge in (table['boundary_lines'] if self.coordinate_mode else []):
                painter.setPen(QPen(QColor('#40C4FF'), 1, Qt.DashLine))
                points = self.canvas_polygon([edge['start'], edge['end']])
                painter.drawLine(points[0], points[1])
        names = {'TOP_LEFT':'左上袋','TOP_MIDDLE':'上中袋','TOP_RIGHT':'右上袋',
                 'BOTTOM_LEFT':'左下袋','BOTTOM_MIDDLE':'下中袋','BOTTOM_RIGHT':'右下袋'}
        states = {'observed':('已验证','#52E377'), 'estimated':('估算','#FFDB4D'), 'missing':('未定位','#FF7070')}
        occupied = []
        font = QFont(self.font());font.setPixelSize(13);font.setBold(True);painter.setFont(font)
        area = self.video_rect()
        for pocket in result['pockets']:
            state, color = states[pocket['status']]
            highlight = QColor(color)
            painter.setPen(QPen(highlight, 2, Qt.SolidLine if pocket['status']=='observed' else Qt.DashLine))
            if pocket['entry_polygon']:
                painter.drawPolygon(self.canvas_polygon(pocket['entry_polygon']))
            marker = pocket['center'] or pocket['candidate_center']
            if marker is None:
                continue
            marker_polygon = self.canvas_polygon([[marker['x'],marker['y']]])
            point = QPointF(marker_polygon[0])
            painter.drawEllipse(point, 6, 6)
            if pocket['status']=='missing':
                painter.drawLine(point+QPointF(-6,-6),point+QPointF(6,6))
                painter.drawLine(point+QPointF(-6,6),point+QPointF(6,-6))
            text = names[pocket['id']] + ' · ' + state
            width = painter.fontMetrics().horizontalAdvance(text)+12
            height = painter.fontMetrics().height()+6
            positions = [(point.x()+9,point.y()-height), (point.x()-width-9,point.y()-height),
                         (point.x()+9,point.y()+9),(point.x()-width-9,point.y()+9)]
            choices=[QRectF(min(max(x,area.left()),max(area.left(),area.right()-width)),
                            min(max(y,area.top()),max(area.top(),area.bottom()-height)),width,height) for x,y in positions]
            label=min(choices,key=lambda box:sum(box.intersects(other) for other in occupied));occupied.append(label)
            painter.fillRect(label,Qt.black);painter.setPen(highlight);painter.drawText(label,Qt.AlignCenter,text)

    def canvas_polygon(self, polygon):
        target = self.video_rect()
        return QPolygonF([QPointF(target.left() + x * target.width() / self.image.width(),
                                 target.top() + y * target.height() / self.image.height())
                          for x, y in polygon])

    def draw_detection(self, painter, box, text, color, font_size, occupied, polygon=None):
        shape = self.canvas_polygon(polygon) if polygon else None
        rect = shape.boundingRect() if shape is not None else self.box_rect(box)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(Qt.black, 5))
        painter.drawPolygon(shape) if shape is not None else painter.drawRect(rect)
        highlight = QColor(color)
        painter.setPen(QPen(highlight, 2))
        painter.drawPolygon(shape) if shape is not None else painter.drawRect(rect)
        font = QFont(self.font())
        font.setPixelSize(font_size)
        font.setBold(True)
        painter.setFont(font)
        width = painter.fontMetrics().horizontalAdvance(text) + 12
        height = painter.fontMetrics().height() + 6
        area = self.video_rect()
        positions = [(rect.left(), rect.top()-height), (rect.left(), rect.bottom()),
                     (rect.right()+4, rect.top()), (rect.left()-width-4, rect.top()),
                     (rect.right()-width, rect.top()-height), (rect.right()-width, rect.bottom())]
        for radius in range(1, 5):
            for dx,dy in [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]:
                positions.append((rect.center().x() + dx*radius*(width+4)-width/2,
                                  rect.center().y() + dy*radius*(height+4)-height/2))
        candidates = [QRectF(min(max(x, area.left()), max(area.left(), area.right()-width)),
                             min(max(y, area.top()), max(area.top(), area.bottom()-height)),
                             width, height) for x,y in positions]
        label = next((candidate for candidate in candidates
                      if not any(candidate.intersects(other) for other in occupied)), None)
        if label is None:
            label = min(candidates, key=lambda candidate: sum(candidate.intersects(other) for other in occupied))
        if not label.intersects(rect.adjusted(-4, -4, 4, 4)):
            end = QPointF(min(max(rect.center().x(), label.left()), label.right()),
                          min(max(rect.center().y(), label.top()), label.bottom()))
            painter.setPen(QPen(highlight, 1))
            painter.drawLine(rect.center(), end)
        occupied.append(label)
        painter.fillRect(label, Qt.black)
        painter.setPen(highlight)
        painter.drawText(label, Qt.AlignCenter, text)


class MainWindow(QMainWindow):
    identify_requested = pyqtSignal(object, float, float, int)
    identify_all_requested = pyqtSignal(object, int)
    live_requested = pyqtSignal(object, int)
    load_requested = pyqtSignal()
    pocket_requested = pyqtSignal(object, int, object)

    def __init__(self, video_path=DEFAULT_VIDEO, image_dir=DEFAULT_IMAGES, start_with_images=False):
        super().__init__()
        self.setWindowTitle("台球视觉识别系统")
        self.resize(1200, 800)
        self.canvas = VideoCanvas()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        gallery = QWidget()
        gallery_layout = QHBoxLayout(gallery)
        gallery_layout.setContentsMargins(16, 8, 16, 4)
        gallery_layout.setSpacing(12)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["视频", "样本图片"])
        self.source_combo.setMinimumSize(120, 44)
        self.source_combo.setAccessibleName("选择视频或样本图片")
        self.previous_button = QPushButton("← 上一张")
        self.next_button = QPushButton("下一张 →")
        for button in (self.previous_button, self.next_button):
            button.setMinimumSize(110, 44)
        self.image_info = QLabel()
        self.image_info.setAlignment(Qt.AlignCenter)
        gallery_layout.addWidget(self.source_combo)
        gallery_layout.addWidget(self.previous_button)
        gallery_layout.addWidget(self.image_info, 1)
        gallery_layout.addWidget(self.next_button)
        layout.addWidget(gallery)
        controls = QWidget()
        buttons = QHBoxLayout(controls)
        buttons.setContentsMargins(16, 8, 16, 8)
        buttons.setSpacing(12)
        self.play_button = QPushButton("播放")
        self.pause_button = QPushButton("暂停")
        self.yolo_button = QPushButton("YOLO：关闭")
        self.yolo_button.setCheckable(True)
        self.yolo_button.setMinimumSize(150, 44)
        self.yolo_button.setToolTip("开启后点选球，再播放只跟踪该球；点击识别全部才识别所有目标")
        buttons.addStretch()
        for button in (self.play_button, self.pause_button):
            button.setMinimumSize(100, 44)
            button.setEnabled(False)
            buttons.addWidget(button)
        buttons.addWidget(self.yolo_button)
        self.identify_all_button = QPushButton("识别全部")
        self.identify_all_button.setMinimumSize(120, 44)
        self.identify_all_button.setEnabled(False)
        self.identify_all_button.setToolTip("播放时开启连续识别；图片或暂停画面识别一次。黑白球分别标名，其余统一标为台球")
        buttons.addWidget(self.identify_all_button)
        self.pocket_button = QPushButton("自动检测袋口")
        self.pocket_button.setMinimumSize(150, 44)
        self.pocket_button.setEnabled(False)
        self.pocket_button.setToolTip("独立检测当前画面的球桌与六袋口，自动保存 JSON 和调试图；可重复检测")
        buttons.addWidget(self.pocket_button)
        buttons.addStretch()
        layout.addWidget(controls)
        coordinate_bar=QWidget()
        coordinate_layout=QHBoxLayout(coordinate_bar)
        coordinate_layout.setContentsMargins(16,0,16,8)
        self.coordinate_button=QPushButton("坐标模式")
        self.coordinate_button.setCheckable(True);self.coordinate_button.setMinimumSize(110,44)
        self.coordinate_button.setToolTip("以内库四角建立毫米坐标；开启时暂停视频，移动鼠标查看坐标，点击球查看球心坐标")
        self.dimensions_button=QPushButton("球桌尺寸")
        self.dimensions_button.setMinimumSize(100,44)
        self.trails_button=QPushButton("轨迹：开启")
        self.trails_button.setCheckable(True);self.trails_button.setChecked(True);self.trails_button.setMinimumSize(100,44)
        self.trails_button.toggled.connect(self.toggle_trails)
        self.trails_button.setToolTip("显示球体固定 ID 和运动轨迹；位置记录自动保存到 data/tracking/")
        coordinate_layout.addWidget(self.trails_button)
        self.coordinate_readout=QLabel("坐标模式未开启")
        self.ball_coordinate_readout=QLabel("球心：未选择")
        self.ball_coordinate_readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        coordinate_layout.addWidget(self.coordinate_button)
        coordinate_layout.addWidget(self.dimensions_button)
        coordinate_layout.addWidget(self.coordinate_readout,1)
        coordinate_layout.addWidget(self.ball_coordinate_readout)
        layout.addWidget(coordinate_bar)
        self.dimension_path=ROOT/'config'/'table_dimensions.json'
        try:
            self.canvas.dimensions=TableDimensions(**json.loads(self.dimension_path.read_text())) if self.dimension_path.exists() else TableDimensions()
        except (ValueError,TypeError,OSError):
            self.canvas.dimensions=TableDimensions()
        self.refresh_dimension_label()
        self.coordinate_button.toggled.connect(self.toggle_coordinates)
        self.dimensions_button.clicked.connect(self.configure_dimensions)
        self.canvas.coordinate_hovered.connect(self.coordinate_readout.setText)
        self.setCentralWidget(content)
        self.capture = None
        self.playback_ready = False
        self.media_mode = "video"
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.image_paths = sorted((p for p in self.image_dir.iterdir()
                                   if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
                                  if self.image_dir.is_dir() else [], key=lambda p: p.name.casefold())
        self.image_index = -1
        self.image_info.setToolTip(str(self.image_dir))
        self.yolo_thread = None
        self.yolo_worker = None
        self.yolo_ready = False
        self.yolo_loading = False
        self.yolo_busy = False
        self.yolo_generation = 0
        self.pending_all = None
        self.track_recorder=TrackRecorder(ROOT/"data"/"tracking")
        self.live_tracking=LiveTracking()
        self.live_frame_id=0
        self.live_token=0
        self.live_inflight=None
        self.live_selected_id=None
        self.selected_targets={}
        self.selection_counter=100000
        self.pending_selection=None
        self.detection_mode="idle"
        self.single_table_polygon=None
        self.live_last_result_time=None
        self.media_generation = 0
        self.pocket_busy = False
        self.pocket_thread = None
        self.pocket_worker = None
        self.last_pocket_result = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.yolo_button.toggled.connect(self.toggle_yolo)
        self.identify_all_button.clicked.connect(self.identify_all_elements)
        self.pocket_button.clicked.connect(self.detect_pockets)
        self.canvas.element_clicked.connect(self.identify_element)
        self.video_path = Path(video_path).expanduser().resolve()
        self.source_combo.currentIndexChanged.connect(self.switch_source)
        self.previous_button.clicked.connect(lambda: self.show_image(self.image_index - 1))
        self.next_button.clicked.connect(lambda: self.show_image(self.image_index + 1))
        self.previous_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.next_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.previous_shortcut.activated.connect(self.previous_button.click)
        self.next_shortcut.activated.connect(self.next_button.click)
        if start_with_images:
            self.source_combo.setCurrentIndex(1)
        else:
            self.switch_source(0)

    def toggle_trails(self,enabled):
        self.canvas.show_trails=enabled
        self.trails_button.setText("轨迹：开启" if enabled else "轨迹：关闭")
        self.canvas.update()

    def sync_tracks(self):
        identities=self.live_tracking.identities
        visible=set(self.selected_targets) if self.detection_mode=="single" else set(identities.states)
        self.canvas.track_paths={key:list(state['path']) for key,state in identities.states.items() if key in visible}
        if self.media_mode!="video" or self.detection_mode=="idle" or self.capture is None or not self.playback_ready:return
        table=next((d for d in self.live_tracking.detections if d['name']=='billiards_table'),None)
        polygon=table.get('playable_polygon') if table else self.single_table_polygon
        coordinates=None
        if polygon:
            try:coordinates=TableCoordinates(polygon,self.canvas.dimensions)
            except (ValueError,cv2.error):pass
        try:
            self.track_recorder.record(identities,self.live_frame_id,
                max(0,int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))-1),self.capture.get(cv2.CAP_PROP_FPS),
                coordinates,str(self.video_path),self.canvas.dimensions.__dict__)
            if self.track_recorder.path:self.trails_button.setToolTip(f"轨迹记录：{self.track_recorder.path}")
        except OSError as exc:
            self.trails_button.setToolTip(f"轨迹显示继续；位置记录写入失败：{exc}")
        self.canvas.update()

    def refresh_dimension_label(self):
        d=self.canvas.dimensions
        prefix="参考" if d.reference else "配置"
        self.dimensions_button.setText(f"{prefix}尺寸：{d.x_mm:g} × {d.y_mm:g} mm")
        self.dimensions_button.setToolTip("内库之间的比赛区域尺寸；X 为上边方向，Y 为左边向下。点击修改并保存。")

    def configure_dimensions(self):
        dialog=QDialog(self);dialog.setWindowTitle("球桌比赛区域尺寸")
        form=QFormLayout(dialog)
        info=QLabel("填写四条内库之间的比赛区域尺寸（mm）。\n原点为左上内库交点，X 向右，Y 向下。\n默认 3569 × 1778 mm 为参考配置，非实测尺寸。")
        form.addRow(info)
        x=QDoubleSpinBox();y=QDoubleSpinBox()
        for spin,value in [(x,self.canvas.dimensions.x_mm),(y,self.canvas.dimensions.y_mm)]:
            spin.setRange(100,20000);spin.setDecimals(1);spin.setSuffix(" mm");spin.setValue(value)
        form.addRow("X 方向尺寸",x);form.addRow("Y 方向尺寸",y)
        actions=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        actions.accepted.connect(dialog.accept);actions.rejected.connect(dialog.reject);form.addRow(actions)
        if dialog.exec_()!=QDialog.Accepted:return
        dimensions=TableDimensions(x.value(),y.value(),False)
        try:
            self.dimension_path.parent.mkdir(parents=True,exist_ok=True)
            temporary=self.dimension_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(dimensions.__dict__,ensure_ascii=False,indent=2))
            temporary.replace(self.dimension_path)
        except OSError as exc:
            self.statusBar().showMessage(f"尺寸保存失败：{exc}")
            return
        self.canvas.dimensions=dimensions;self.canvas.coordinates=None;self.canvas.coordinate_ball=None
        self.refresh_dimension_label()
        self.ball_coordinate_readout.setText("球心：未选择")
        if self.canvas.coordinate_mode:self.canvas.calibrate_coordinates()
        self.canvas.update()

    def toggle_coordinates(self,enabled):
        self.canvas.coordinate_mode=enabled
        self.coordinate_button.setText("坐标模式：开启" if enabled else "坐标模式")
        self.ball_coordinate_readout.setText("球心：未选择")
        if enabled:
            self.canvas.calibrate_coordinates()
            if not self.yolo_button.isChecked():self.yolo_button.setChecked(True)
        else:
            self.canvas.coordinates=None;self.canvas.coordinate_ball=None
            self.coordinate_readout.setText("坐标模式未开启")
        self.canvas.update()

    def invalidate_results(self):
        self.track_recorder.close();self.canvas.track_paths={}
        self.selected_targets.clear();self.canvas.marked_ids.clear();self.pending_selection=None
        self.detection_mode="idle"
        self.live_tracking.clear()
        self.live_selected_id=None
        self.canvas.coordinates=None
        self.canvas.coordinate_ball=None
        if self.canvas.coordinate_mode:self.coordinate_readout.setText("等待当前画面内库定位")
        self.ball_coordinate_readout.setText("球心：未选择")
        self.media_generation += 1
        self.last_pocket_result = None
        self.canvas.clear_pockets()
        self.yolo_generation += 1
        self.pending_all = None
        self.canvas.clear_selection()

    def update_navigation(self):
        images = self.media_mode == "images"
        self.previous_button.setEnabled(images and self.image_index > 0)
        self.next_button.setEnabled(images and self.image_index + 1 < len(self.image_paths))
        self.previous_shortcut.setEnabled(images)
        self.next_shortcut.setEnabled(images)
        if images and self.image_index >= 0:
            self.image_info.setText(f"{self.image_index + 1} / {len(self.image_paths)} · {self.image_paths[self.image_index].name}")
        elif images:
            self.image_info.setText("未找到样本图片")
        else:
            self.image_info.setText(self.video_path.name)
        self.play_button.setEnabled(not images and self.playback_ready and not self.timer.isActive())
        self.pause_button.setEnabled(not images and self.playback_ready and self.timer.isActive())
        self.update_all_button()

    def switch_source(self, index):
        self.timer.stop()
        self.invalidate_results()
        self.media_mode = "images" if index == 1 else "video"
        self.playback_ready = False
        if self.media_mode == "images":
            if self.image_paths:
                self.show_image(max(0, self.image_index))
            else:
                self.canvas.frame = None
                self.canvas.image = QImage()
                self.show_error(f"未找到样本图片：{self.image_dir}")
                self.update_navigation()
            return
        if self.capture is None or not self.capture.isOpened():
            if not self.video_path.is_file():
                self.canvas.image = QImage()
                self.show_error(f"未找到视频：{self.video_path}\n请先运行 scripts/download_demo_assets.py。")
                self.update_navigation()
                return
            self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            self.canvas.image = QImage()
            self.show_error(f"无法打开视频：{self.video_path.name}")
            self.update_navigation()
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.next_frame()
        if not self.canvas.image.isNull():
            self.playback_ready = True
            self.timer.setInterval(round(1000 / fps) if 1 <= fps <= 120 else 40)
            self.play_video()
        self.update_navigation()

    def show_image(self, index):
        if self.media_mode != "images" or not 0 <= index < len(self.image_paths):
            return
        self.timer.stop()
        self.invalidate_results()
        self.image_index = index
        path = self.image_paths[index]
        try:
            frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("图片无法解码")
        except (OSError, ValueError, cv2.error) as exc:
            self.canvas.frame = None
            self.canvas.image = QImage()
            self.show_error(f"无法打开图片 {path.name}：{exc}")
        else:
            self.canvas.set_frame(frame)
            self.playback_ready = True
            self.statusBar().showMessage(f"样本图片：{path.name} · 点击“识别全部”或开启 YOLO 后点选")
        self.update_navigation()

    def toggle_yolo(self, enabled):
        self.ball_coordinate_readout.setText("球心：未选择")
        self.yolo_generation += 1
        self.live_last_result_time=None
        self.canvas.selection_enabled = enabled
        self.canvas.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.yolo_button.setText("YOLO：开启" if enabled else "YOLO：关闭")
        if not enabled:
            self.track_recorder.close();self.canvas.track_paths={}
            self.selected_targets.clear();self.canvas.marked_ids.clear();self.pending_selection=None
            self.detection_mode="idle"
            self.live_tracking.clear()
            self.live_selected_id=None
            self.pending_all = None
            self.update_all_button()
            self.canvas.clear_selection()
            self.statusBar().showMessage("YOLO 已关闭")
            return
        if self.yolo_ready:
            self.maybe_submit_live()
            self.statusBar().showMessage("YOLO 实时识别已开启，视频继续播放" if self.live_active() else
                                         "正在完成上一次推理…" if self.yolo_busy else "YOLO 已开启，点击画面中的物体")
            return
        self.statusBar().showMessage("正在加载 YOLO，视频可以继续播放")
        if self.yolo_thread is None:
            self.yolo_thread = QThread(self)
            self.yolo_worker = YoloWorker()
            self.yolo_worker.moveToThread(self.yolo_thread)
            self.yolo_thread.finished.connect(self.yolo_worker.deleteLater)
            self.identify_requested.connect(self.yolo_worker.identify)
            self.identify_all_requested.connect(self.yolo_worker.identify_all)
            self.live_requested.connect(self.yolo_worker.identify_live)
            self.load_requested.connect(self.yolo_worker.load)
            self.yolo_worker.ready.connect(self.on_yolo_ready)
            self.yolo_worker.result.connect(self.on_yolo_result)
            self.yolo_worker.all_result.connect(self.on_yolo_all_result)
            self.yolo_worker.live_result.connect(self.on_live_result)
            self.yolo_worker.live_failed.connect(self.on_live_error)
            self.yolo_worker.failed.connect(self.on_yolo_error)
            self.yolo_thread.start()
        if not self.yolo_loading:
            self.yolo_loading = True
            self.update_all_button()
            self.load_requested.emit()

    def on_yolo_ready(self):
        self.yolo_ready = True
        self.yolo_loading = False
        self.update_all_button()
        if self.pending_all is not None:
            frame, generation = self.pending_all
            self.pending_all = None
            if self.yolo_button.isChecked() and generation == self.yolo_generation:
                self.submit_all(frame, generation)
                return
        self.maybe_submit_live()
        if self.yolo_button.isChecked() and not self.live_active():
            self.statusBar().showMessage("YOLO 已开启，点击视频中的物体；可暂停后选择")

    def live_active(self):
        return self.media_mode=="video" and self.timer.isActive() and self.yolo_button.isChecked() and self.detection_mode!="idle"

    def maybe_submit_live(self):
        if not self.live_active() or not self.yolo_ready or self.yolo_loading or self.yolo_busy or self.canvas.frame is None:
            return
        if not self.live_tracking.frames:
            self.live_tracking.push(self.live_frame_id,self.canvas.frame)
        self.live_token+=1
        self.live_inflight=(self.live_token,self.yolo_generation,self.live_frame_id,time.monotonic())
        self.yolo_busy=True
        self.update_all_button()
        payload=self.canvas.frame.copy()
        if self.detection_mode=="single":
            targets=list(self.selected_targets.values())
            if not targets:
                self.yolo_busy=False;self.live_inflight=None;self.update_all_button();return
            payload={"frame":payload,"target":targets[0]} if len(targets)==1 else {"frame":payload,"targets":targets}
        self.live_requested.emit(payload,self.live_token)

    def on_live_result(self,token,detections):
        request=self.live_inflight
        if request is None or token!=request[0]:return
        self.live_inflight=None;self.yolo_busy=False
        self.update_all_button()
        if self.pending_selection is not None:
            frame,x,y,epoch=self.pending_selection;self.pending_selection=None
            if epoch==self.yolo_generation:self.identify_element(frame,x,y)
            return
        if not self.yolo_button.isChecked() or request[1]!=self.yolo_generation:
            self.maybe_submit_live();return
        # Never apply a result older than the bounded frame history.
        if self.live_tracking.replace(request[2],detections):
            if self.detection_mode=="single":
                self.live_tracking.detections=[item for item in self.live_tracking.detections if item.get("track_id") in self.selected_targets]
            self.canvas.set_detections(self.live_tracking.detections)
            self.render_live_selection()
            self.sync_tracks()
            now=time.monotonic()
            rate=1/(now-self.live_last_result_time) if self.live_last_result_time else 0.
            self.live_last_result_time=now
            suffix=f" · 推理更新 {rate:.1f} 次/秒" if rate else ""
            self.statusBar().showMessage(f"YOLO 实时识别：{len(detections)} 个目标{suffix} · 播放与推理独立")
        # The next video tick submits only the newest frame; no FIFO of images.

    def on_live_error(self,token,message):
        request=self.live_inflight
        if request is None or token!=request[0]:return
        self.live_inflight=None;self.yolo_busy=False
        self.live_tracking.detections=[];self.canvas.clear_selection()
        if self.yolo_worker and self.yolo_worker.process and self.yolo_worker.process.poll() is not None:
            self.yolo_ready=False
        self.update_all_button()
        if request[1]==self.yolo_generation:self.statusBar().showMessage(f"YOLO 实时识别失败：{message}")

    def toggle_target(self,item):
        from src.table_geometry import contains_point
        box=item['box'];cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        existing=next((key for key,target in self.selected_targets.items()
                       if target['name']==item['name'] and contains_point(target,cx,cy)),None)
        if existing is not None:
            del self.selected_targets[existing]
        else:
            key=item.get('track_id') if item.get('track_id') in self.live_tracking.identities.states else None
            if key is None:
                nearby=[(float(np.linalg.norm(state['center']-np.array([cx,cy]))),old_key) for old_key,state in self.live_tracking.identities.states.items()
                        if state['name']==item['name'] and old_key not in self.selected_targets
                        and np.linalg.norm(state['center']-np.array([cx,cy]))<.75*max(box[2]-box[0],box[3]-box[1])]
                if nearby:key=min(nearby)[1]
            if key is None:self.selection_counter+=1;key=self.selection_counter
            self.selected_targets[key]={**item,'selection_id':key,'track_id':key}
        self.yolo_generation+=1
        self.detection_mode="single" if self.selected_targets else "idle"
        self.live_selected_id=next(reversed(self.selected_targets),None)
        self.canvas.marked_ids=set(self.selected_targets)
        self.live_tracking.clear(keep_tracks=True)
        self.live_tracking.frames=deque(maxlen=64 if len(self.selected_targets)>1 else 16)
        if self.canvas.frame is not None:
            self.live_tracking.push(self.live_frame_id,self.canvas.frame)
            self.live_tracking.replace(self.live_frame_id,list(self.selected_targets.values()))
        self.canvas.set_detections(self.live_tracking.detections or list(self.selected_targets.values()))
        self.render_live_selection()
        self.sync_tracks()
        if self.canvas.frame is None and self.selected_targets:
            item=self.selected_targets[self.live_selected_id]
            self.canvas.set_selection(item["box"],f"{chinese_label(item['name'])} {item['confidence']:.0%}",item.get("polygon"))
        self.update_all_button()
        self.statusBar().showMessage(f"已选 {len(self.selected_targets)} 个目标 · 再点取消，点其他目标可多选")
        if not self.selected_targets:self.ball_coordinate_readout.setText("未选择目标")

    def render_live_selection(self):
        for item in self.live_tracking.detections:
            if item.get('track_id') in self.selected_targets:self.selected_targets[item['track_id']]=item
        self.canvas.marked_ids=set(self.selected_targets)
        if self.live_selected_id is None:return
        item=next((d for d in self.live_tracking.detections if d.get('track_id')==self.live_selected_id),None)
        if item is None:
            if self.detection_mode!="single":self.live_selected_id=None
            self.ball_coordinate_readout.setText("球心：所选目标暂时丢失");return
        label=chinese_label(item['name'])
        self.canvas.selected_box=item['box'];self.canvas.selected_polygon=item.get('polygon')
        identity=f" B{item['ball_id']}" if item.get("ball_id") else ""
        self.canvas.selected_label=f"{label}{identity} {item['confidence']:.0%}"
        if item['name']!='billiards_table':
            table=next((d for d in self.live_tracking.detections if d['name']=='billiards_table'),None)
            if table and table.get('playable_polygon'):
                try:self.canvas.coordinates=TableCoordinates(table['playable_polygon'],self.canvas.dimensions)
                except (ValueError,cv2.error):self.canvas.coordinates=None
            elif self.single_table_polygon:
                self.canvas.coordinates=TableCoordinates(self.single_table_polygon,self.canvas.dimensions)
            position=self.canvas.coordinates.ball_center(item['box']) if self.canvas.coordinates else None
            self.canvas.coordinate_ball=position
            self.ball_coordinate_readout.setText(f"已选 {len(self.selected_targets)} 个 | {label}：X = {position[0]:.1f} mm，Y = {position[1]:.1f} mm" if position else f"已选 {len(self.selected_targets)} 个目标")
        if item["name"]=="billiards_table":self.ball_coordinate_readout.setText(f"已选 {len(self.selected_targets)} 个（当前：台球桌）")
        self.canvas.update()

    def update_pocket_button(self):
        self.pocket_button.setText("袋口检测中…" if self.pocket_busy else "自动检测袋口")
        self.pocket_button.setEnabled(self.playback_ready and not self.pocket_busy)

    def detect_pockets(self):
        if not self.playback_ready or self.canvas.frame is None or self.pocket_busy:
            return
        self.pause_video()
        frame = self.canvas.frame.copy()
        self.canvas.clear_pockets()
        self.last_pocket_result = None
        self.pocket_busy = True
        self.update_pocket_button()
        self.statusBar().showMessage("正在独立检测球桌和袋口，并保存调试结果…")
        if self.pocket_thread is None:
            self.pocket_thread = QThread(self)
            self.pocket_worker = PocketWorker()
            self.pocket_worker.moveToThread(self.pocket_thread)
            self.pocket_thread.finished.connect(self.pocket_worker.deleteLater)
            self.pocket_requested.connect(self.pocket_worker.detect)
            self.pocket_worker.done.connect(self.on_pocket_result)
            self.pocket_worker.failed.connect(self.on_pocket_error)
            self.pocket_thread.start()
        if self.media_mode == 'images':
            source = {'kind':'image', 'path':str(self.image_paths[self.image_index])}
        else:
            source = {'kind':'video', 'path':str(self.video_path),
                      'frame_index':max(0,int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))-1)}
        self.pocket_requested.emit(frame, self.media_generation, source)

    def on_pocket_result(self, generation, result):
        self.pocket_busy = False
        self.update_pocket_button()
        if generation != self.media_generation:
            return
        self.last_pocket_result = result
        self.canvas.set_pocket_result(result)
        counts = {state:sum(p['status']==state for p in result['pockets']) for state in ('observed','estimated','missing')}
        state = "六袋口验证通过" if result['valid'] else "定位不完整"
        reasons = '、'.join(result['reasons'][:2])
        message = f"{state}：已验证 {counts['observed']}，估算 {counts['estimated']}，未定位 {counts['missing']} · {result['elapsed_ms']:.0f} ms"
        if not result['valid']:
            message += " · 部分袋口被遮挡或可见特征不足"
        path = result['artifacts'].get('json','')
        self.pocket_button.setToolTip(f"再次点击重新检测。上次结果：{path}\n原因：{reasons}")
        self.statusBar().showMessage(message + f" · JSON：{Path(path).parent.name}/result.json")

    def on_pocket_error(self, generation, message):
        self.pocket_busy = False
        self.update_pocket_button()
        if generation == self.media_generation:
            self.statusBar().showMessage(f"袋口检测或保存失败：{message}")

    def update_all_button(self):
        self.update_pocket_button()
        text = "加载中…" if self.yolo_loading else "识别全部" if self.detection_mode=="single" else "实时识别中" if self.live_active() else "识别中…" if self.yolo_busy else "识别全部"
        self.identify_all_button.setText(text)
        self.identify_all_button.setEnabled(self.playback_ready and not self.yolo_loading and (not self.yolo_busy or self.live_active()))

    def identify_all_elements(self):
        self.ball_coordinate_readout.setText("球心：未选择")
        if not self.playback_ready or self.canvas.frame is None or (self.yolo_busy and not self.live_active()):
            return
        if self.detection_mode!="all":self.yolo_generation+=1
        self.detection_mode="all"
        self.live_tracking.frames=deque(self.live_tracking.frames,maxlen=16)
        self.selected_targets.clear();self.canvas.marked_ids.clear()
        self.live_selected_id=None
        if self.media_mode=="video" and self.timer.isActive():
            if not self.yolo_button.isChecked():self.yolo_button.setChecked(True)
            self.maybe_submit_live()
            return
        frame = self.canvas.frame.copy()
        self.canvas.set_frame(frame, keep_pockets=True)
        if not self.yolo_button.isChecked():
            self.yolo_button.setChecked(True)
        elif not self.yolo_ready and not self.yolo_loading:
            self.toggle_yolo(True)
        if self.yolo_ready:
            self.submit_all(frame, self.yolo_generation)
        else:
            self.pending_all = (frame, self.yolo_generation)
            self.statusBar().showMessage("正在加载 YOLO，随后识别已暂停的当前画面…")

    def submit_all(self, frame, generation):
        self.yolo_busy = True
        self.update_all_button()
        self.statusBar().showMessage("正在识别当前画面中的全部目标…")
        self.identify_all_requested.emit(frame, generation)

    def on_yolo_all_result(self, generation, detections):
        self.yolo_busy = False
        self.update_all_button()
        if not self.yolo_button.isChecked() or generation != self.yolo_generation:
            return
        self.canvas.set_detections(detections)
        if detections:
            counts = {}
            for item in detections:
                name = chinese_label(item["name"])
                counts[name] = counts.get(name, 0) + 1
            summary = "、".join(f"{name} {count} 个" for name,count in counts.items())
            self.statusBar().showMessage(f"检出 {len(detections)} 个目标：{summary} · 已暂停，点击播放继续")
        else:
            self.statusBar().showMessage("当前画面未可靠定位台球桌，请换一帧重试")

    def identify_element(self, frame, x, y):
        from src.table_geometry import contains_point
        # Toggling a known selection does not wait for an inference request.
        matches=[d for d in self.selected_targets.values() if contains_point(d,x,y)]
        if not matches:matches=[d for d in self.canvas.detections if contains_point(d,x,y)]
        if matches and min(matches,key=lambda d:(d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]))['name']!='billiards_table':
            item=min(matches,key=lambda d:(d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]))
            if self.canvas.coordinates:self.single_table_polygon=self.canvas.coordinates.corners.tolist()
            self.toggle_target(item);return
        if not self.yolo_button.isChecked() or not self.yolo_ready:return
        self.pause_video()
        if self.yolo_busy:
            self.pending_selection=(frame.copy(),x,y,self.yolo_generation)
            return
        self.yolo_busy=True;self.update_all_button()
        self.ball_coordinate_readout.setText(f"已选 {len(self.selected_targets)} 个 · 正在识别新目标…")
        self.identify_requested.emit(frame,x,y,self.yolo_generation)

    def on_yolo_result(self, generation, name, confidence, box=None, polygon=None):
        self.yolo_busy = False
        self.update_all_button()
        if not self.yolo_button.isChecked():
            return
        if generation != self.yolo_generation:
            self.statusBar().showMessage("YOLO 已开启，点击视频中的物体；可暂停后选择")
            return
        if name and box is not None:
            if self.canvas.coordinates is None:self.canvas.calibrate_coordinates()
            if self.canvas.coordinates:self.single_table_polygon=self.canvas.coordinates.corners.tolist()
            self.toggle_target({'name':name,'confidence':confidence,'box':box,'polygon':polygon})
        else:
            self.ball_coordinate_readout.setText(f"已选 {len(self.selected_targets)} 个 · 点击处未识别到目标")

    def on_yolo_error(self, generation, message):
        self.yolo_busy = False
        self.pending_all = None
        if generation == -1:
            self.yolo_loading = False
            self.yolo_ready = False
        elif self.yolo_worker.process is None or self.yolo_worker.process.poll() is not None:
            self.yolo_ready = False
        self.update_all_button()
        if self.yolo_button.isChecked() and (generation == -1 or generation == self.yolo_generation):
            self.statusBar().showMessage(f"YOLO 错误：{message}")
        elif self.yolo_button.isChecked():
            self.statusBar().showMessage("YOLO 已开启，点击视频中的物体；可暂停后选择")

    def play_video(self):
        self.ball_coordinate_readout.setText("球心：未选择")
        if self.media_mode == "images" or not self.playback_ready:
            return
        self.yolo_generation += 1
        self.media_generation += 1
        self.last_pocket_result = None
        self.canvas.clear_pockets()
        self.pending_all = None
        self.update_all_button()
        self.canvas.clear_selection()
        if self.detection_mode!="single":
            self.live_tracking.clear();self.live_selected_id=None
        self.timer.start()
        self.maybe_submit_live()
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.statusBar().showMessage(f"本地视频：{self.video_path.name} · 正在循环播放")

    def pause_video(self):
        if self.media_mode == "images" or not self.playback_ready:
            return
        self.timer.stop()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.statusBar().showMessage(f"本地视频：{self.video_path.name} · 已暂停")

    def show_error(self, message):
        self.timer.stop()
        self.playback_ready = False
        self.pending_all = None
        self.update_all_button()
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.canvas.message = message
        self.canvas.update()
        self.statusBar().showMessage("图片加载失败" if self.media_mode == "images" else "视频播放失败")
        print(message, file=sys.stderr, flush=True)

    def next_frame(self):
        self.ball_coordinate_readout.setText("球心：未选择")
        self.media_generation += 1
        self.last_pocket_result = None
        ok, frame = self.capture.read()
        if not ok:
            self.yolo_generation+=1
            self.live_tracking.clear();self.live_selected_id=None
            if self.detection_mode=="single":
                self.detection_mode="idle";self.selected_targets.clear();self.canvas.marked_ids.clear()
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if not ok:
            self.canvas.image = QImage()
            self.show_error("无法读取视频帧，请检查视频文件是否完整。")
            return
        self.live_frame_id+=1
        tracked=self.live_tracking.push(self.live_frame_id,frame) if self.yolo_button.isChecked() else []
        self.canvas.set_frame(frame)
        if self.yolo_button.isChecked():
            self.canvas.set_detections(tracked)
            self.render_live_selection()
            self.sync_tracks()
            self.maybe_submit_live()

    def closeEvent(self, event):
        self.track_recorder.close()
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
        if self.yolo_thread is not None:
            if self.yolo_worker.process is not None:
                self.yolo_worker.process.terminate()
            self.yolo_thread.quit()
            self.yolo_thread.wait()
            self.yolo_worker.shutdown()
        if self.pocket_thread is not None:
            self.pocket_thread.requestInterruption()
            self.pocket_thread.quit()
            self.pocket_thread.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="播放本地台球视频或样本图片")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--source", choices=["images", "video"])
    args = parser.parse_args()
    app = QApplication([sys.argv[0]])
    app.setApplicationName("台球视觉识别")
    app.setWindowIcon(QIcon(str(ROOT / "assets" / "macos" / "logo.png")))
    source = args.source or ("video" if args.video else "images")
    window = MainWindow(args.video or DEFAULT_VIDEO, args.images_dir, start_with_images=source == "images")
    window.show()
    sys.exit(app.exec_())

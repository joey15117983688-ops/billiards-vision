"""Play a local snooker match recording in the existing PyQt5 window."""
# 桌面入口：主线程负责 Qt 事件与视频解码，耗时校准通过 QThread 执行，YOLO 通过独立进程执行。
# 几何约定：playable_polygon 是内库比赛平面；recognition_polygon/scope_polygon 还可包含相连库边。
# 异步结果必须核对 generation/token；换源、重播或切换模式后，旧请求不得更新当前画面。
# 此文件只是协调现有观测与状态机；预测轨迹不应被理解为真实入袋证据。

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import hashlib
from copy import deepcopy
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
    QApplication, QComboBox, QInputDialog, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QMainWindow, QPushButton, QShortcut, QVBoxLayout, QWidget,
)
from src.yolo_labels import chinese_label
from src.table_coordinates import TableCoordinates, TableDimensions
from src.live_tracking import LiveTracking
from src.track_recording import TrackRecorder

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "data" / "test.mp4"
if not DEFAULT_VIDEO.is_file():DEFAULT_VIDEO = ROOT / "data" / "overhead_pool.mp4"
DEFAULT_IMAGES = ROOT / "data" / "dataset" / "images" / "train"


# Qt 与模型子进程之间的通信桥；信号将结果交回 UI，不直接操作界面控件。
class YoloWorker(QObject):
    geometry_ready = pyqtSignal(int, object)
    ready = pyqtSignal()
    result = pyqtSignal(int, str, float, object, object)
    all_result = pyqtSignal(int, object)
    live_result = pyqtSignal(int, object)
    live_failed = pyqtSignal(int, str)
    failed = pyqtSignal(int, str)

    # 保存子进程、日志句柄和当前几何；模型不会在构造时加载，由 load 信号显式触发。
    def __init__(self):
        super().__init__()
        self.process = None
        self.log = None
        self.calibration_polygon = None
        self.scope_polygon = None

    # 读取一行 JSON 协议消息，EOF 或 error 字段转成调用方可显示的异常。
    # 这里仍是阻塞 readline，并没有通信超时保护；注释不代表该风险已经修复。
    def read_reply(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("YOLO 推理进程已退出，请重新开启 YOLO；详情见 logs/yolo_ui.log")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    # 先回收旧进程，再启动当前 Python 环境中的模型 worker，并等待 ready 握手。
    # stderr 写本地诊断日志，stdout 专用于 JSON 协议，避免模型日志破坏解析。
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

    # 将当前帧存成禁用 pickle 的临时 NPY，再通过 JSON 发送文件位置与请求选项。
    # 无论成功或异常都删除临时帧；返回值是 worker 的真实响应，不补造检测结果。
    def request_prediction(self, frame, options):
        path = None
        try:
            with tempfile.NamedTemporaryFile(dir=ROOT / "logs", suffix=".npy", delete=False) as f:
                np.save(f, frame, allow_pickle=False)
                path = Path(f.name)
            self.process.stdin.write(json.dumps({"frame": str(path), "calibration_polygon": self.calibration_polygon, "scope_polygon": self.scope_polygon or self.calibration_polygon, **options}) + "\n")
            self.process.stdin.flush()
            return self.read_reply()
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    # 对点击当时的原始图像执行点选请求；几何和检测结果都携带同一世代号返回。
    @pyqtSlot(object, float, float, int)
    def identify(self, frame, x, y, generation):
        try:
            reply = self.request_prediction(frame, {"x": x, "y": y})
            geometry=reply.get("scene_geometry") or (reply if reply.get("calibration_evidence") else None)
            if geometry is not None:self.geometry_ready.emit(generation,geometry)
            self.result.emit(generation, reply["name"], reply["confidence"], reply.get("box"), reply.get("polygon"))
        except Exception as exc:
            self.failed.emit(generation, str(exc))

    # 静态整帧识别入口；全目标列表通过 all_result 返回，异常仍绑定该次请求世代。
    @pyqtSlot(object, int)
    def identify_all(self, frame, generation):
        try:
            reply = self.request_prediction(frame, {"mode": "all"})
            self.all_result.emit(generation, reply["detections"])
        except Exception as exc:
            self.failed.emit(generation, str(exc))

    # 根据载荷是否含 targets/target 切换多选局部、单选局部或整帧推理。
    # token 交给主窗口核对，保证延迟响应不会串入新的选择或来源。
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

    # 关闭时先 terminate，三秒内不退出再 kill，随后回收管道和日志句柄。
    # 该等待上界只适用于这里的子进程退出步骤，不等于整个 UI 关闭流程有超时。
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


# 单帧袋口检测任务；与 YOLO 开关独立，输出几何状态而非进球事件。
class PocketWorker(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    # 用传入的冻结帧检测台面和袋口；任务被中断时不保存或发布后续结果。
    # 调试图片仅写本地 data/debug，完成信号带 generation 供主窗口忽略旧结果。
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


# 显式校准任务：重新读取源帧、搜索几何并回传；不会逐帧自动覆盖保存基准。
class CalibrationWorker(QObject):
    progress = pyqtSignal(int, str)
    done = pyqtSignal(int, object, object)
    failed = pyqtSignal(int, str)

    # 按来源重新读取指定图片或视频帧，读取失败就报错，不以旧画面冒充新输入。
    # 已有人工样本收集只在请求显式携带标注时执行；geometry_only 避免为校准额外运行球体模型。
    @pyqtSlot(object, int, object)
    def run(self, fallback, generation, source):
        try:
            from src.scene_calibration import calibrate_scene
            self.progress.emit(generation, "1/4 重新读取当前画面…")
            if source['kind']=='image':
                frame=cv2.imread(source['path'])
            else:
                capture=cv2.VideoCapture(source['path'])
                try:
                    capture.set(cv2.CAP_PROP_POS_FRAMES,source['frame_index'])
                    ok,frame=capture.read()
                    if not ok:frame=None
                finally:
                    capture.release()
            if frame is None:
                raise ValueError('重新读取画面失败，请检查视频或图片来源')
            sample_path=None;sample_error=None
            if source.get('collect_training_sample') and source.get('manual_profile') is not None:
                try:
                    from src.calibration_samples import save_calibration_sample
                    metadata={key:value for key,value in source.items() if key not in ('manual_profile','collect_training_sample')}
                    sample_path=str(save_calibration_sample(ROOT/'data'/'calibration_samples',frame,source['manual_profile'],metadata))
                except (OSError,ValueError,TypeError,cv2.error) as exc:
                    sample_error=str(exc)
            result=calibrate_scene(frame, None,
                                   lambda message:self.progress.emit(generation,message),source.get("manual_profile"),
                                   source.get("correction_context"), geometry_only=True)
            result['learning_sample']=sample_path
            result['learning_sample_error']=sample_error
            result['pockets']['source']=source
            self.done.emit(generation,frame,result)
        except Exception as exc:
            self.failed.emit(generation,str(exc))


# 原图显示及叠加层；所有识别数据保留原始图像坐标，绘制/点选时才映射到控件坐标。
class VideoCanvas(QWidget):
    fullscreen_requested = pyqtSignal()
    pockets_available = pyqtSignal(bool)
    element_clicked = pyqtSignal(object, float, float)
    coordinate_hovered = pyqtSignal(str)

    # 初始化独立的几何、选择、轨迹和袋口叠加状态；旋转仅影响显示，不修改原始帧。
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rotate_portrait=False
        self.fullscreen_button=QPushButton("旋转放大",self)
        self.fullscreen_button.setMinimumSize(110,44)
        self.fullscreen_button.setToolTip("竖屏顺时针旋转并适配当前视频区域；再次点击或按 Esc 恢复")
        self.fullscreen_button.clicked.connect(self.fullscreen_requested)
        self.image = QImage()
        self.frame = None
        self.calibration_polygon = None
        self.recognition_polygon = None
        self.exam_circles=[]
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
        self.show_trails=False
        self.show_ball_boxes=True
        self.show_pocket_overlays=True
        self.marked_ids=set()
        self.message = "正在打开视频…"
        self.setMinimumSize(320, 240)
        self.setAccessibleName("台球视频或样本图片，可开启 YOLO 后点击识别物体")

    # 按 keep_pockets 决定是否保留袋口，清理旧选择并将 BGR 图像转为 Qt 图像。
    # 视频连续播放时由调用方显式保留固定几何，不能把每次绘制当成重新校准。
    def set_frame(self, frame, keep_pockets=False):
        if not keep_pockets:
            self.pocket_result = None
            self.pockets_available.emit(False)
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

    # 接受袋口结果并同步画布几何/坐标可用状态，通知界面更新相关按钮。
    def set_pocket_result(self, result):
        self.pocket_result = result
        self.pockets_available.emit(result is not None)
        if self.coordinate_mode:
            self.calibrate_coordinates()
        self.update()

    # 清空袋口覆盖层并通知控件；与已经写入磁盘的校准配置分开管理。
    def clear_pockets(self):
        self.pocket_result = None
        self.pockets_available.emit(False)
        self.update()

    # 删除当前高亮目标及球心读数，避免把上一帧的点选结果当作当前观测。
    def clear_selection(self):
        self.coordinate_ball = None
        self.selected_box = None
        self.selected_polygon = None
        self.selected_label = ""
        self.detections = []
        self.update()

    # 保存一次点选目标的名称、框和可选多边形，供绘制及坐标读取使用。
    def set_selection(self, box, label, polygon=None):
        self.detections = []
        self.selected_box = box
        self.selected_polygon = polygon
        self.selected_label = label
        self.update()

    # 更新整组检测并尝试保留可匹配的选择；毫米球心取检测框中心，不取鼠标点。
    def set_detections(self, detections):
        self.clear_selection()
        self.detections = [dict(d) for d in detections]
        table=next((d for d in self.detections if d['name']=='billiards_table' and d.get('polygon')),None)
        if table:
            if self.calibration_polygon is not None:
                # Only explicit calibration/source changes may replace a validated
                # table. A player's arm is not evidence that the table moved.
                polygon=self.table_outline()
                table['polygon']=polygon
                points=np.asarray(polygon,float)
                table['box']=[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]
            else:
                self.recognition_polygon=table['polygon']
                if self.pocket_result:self.pocket_result['table']['scope_polygon']=table['polygon']
        if self.coordinate_mode:self.calibrate_coordinates()
        self.update()

    # 将选中框转为控件绘制矩形；无选择时不产生虚构的矩形。
    def selected_rect(self):
        return self.box_rect(self.selected_box)

    # 只有启用旋转且原图为竖向时才旋转，横向视频不再次旋转。
    def is_rotated(self):
        return self.rotate_portrait and not self.image.isNull() and self.image.height()>self.image.width()

    # 窗口尺寸变化时重定位右上角旋转按钮，叠加坐标仍由当前显示矩形计算。
    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.fullscreen_button.adjustSize()
        self.fullscreen_button.move(max(0,self.width()-self.fullscreen_button.width()-12),12)
        self.fullscreen_button.raise_()

    # 返回旋转后的逻辑图像宽高，供等比例缩放使用。
    def display_dimensions(self):
        return (self.image.height(),self.image.width()) if self.is_rotated() else (self.image.width(),self.image.height())

    # 将原始像素点映射到当前显示矩形；竖屏旋转与黑边偏移在此统一处理。
    def display_point(self,x,y):
        if self.is_rotated():x,y=self.image.height()-y,x
        width,height=self.display_dimensions();area=self.video_rect()
        return QPointF(area.left()+x*area.width()/max(1,width),area.top()+y*area.height()/max(1,height))

    # display_point 的反向映射；点选必须用原图坐标提交给检测器。
    def image_point(self,point):
        area=self.video_rect()
        if self.image.isNull() or not area.contains(point) or area.isEmpty():return None
        width,height=self.display_dimensions()
        x=(point.x()-area.left())*width/area.width()
        y=(point.y()-area.top())*height/area.height()
        return (y,self.image.height()-x) if self.is_rotated() else (x,y)

    # 映射框的端点并规范化矩形，处理旋转后左右/上下端点次序变化。
    def box_rect(self, box):
        if box is None or self.image.isNull():return QRectF()
        x1,y1,x2,y2=box
        return self.canvas_polygon([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]).boundingRect()

    # 等比例适配图像并居中，黑边不属于视频内容或可点选台面。
    def video_rect(self):
        from PyQt5.QtCore import QSize
        width,height=self.display_dimensions()
        size=QSize(width,height).scaled(self.size(),Qt.KeepAspectRatio)
        return QRectF((self.width()-size.width())/2,(self.height()-size.height())/2,size.width(),size.height())

    # 按已校准几何与现有检测选择可用台面；不是每次绘制都重新调用检测器。
    def table_outline(self):
        if self.calibration_polygon is not None:
            if self.recognition_polygon is not None:return self.recognition_polygon
            if self.pocket_result:
                return self.pocket_result['table'].get('scope_polygon') or self.calibration_polygon
            return self.calibration_polygon
        table=next((d for d in self.detections if d['name']=='billiards_table'),None)
        if table and table.get('polygon'):return table['polygon']
        if self.selected_polygon is not None and self.selected_label.startswith('台球桌'):return self.selected_polygon
        if self.recognition_polygon:return self.recognition_polygon
        if self.pocket_result:
            table=self.pocket_result['table']
            return table.get('scope_polygon') or table.get('playable_polygon')
        return self.calibration_polygon

    # 用可靠内库四边形建立毫米映射，边界或透视不可靠时返回失败。
    def calibrate_coordinates(self):
        self.coordinates = None
        self.coordinate_ball = None
        if self.frame is None:
            self.coordinate_hovered.emit("尚无画面")
            return
        from src.target_balls import playing_surface
        polygon = self.table_outline()
        if polygon is None and self.pocket_result and self.pocket_result['table']['quality'].get('geometry_reliable'):
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

    # 将悬停位置换算为原图/毫米坐标；黑边、台外或无可靠映射时不显示伪坐标。
    def mouseMoveEvent(self, event):
        if self.coordinate_mode:
            area=self.video_rect();point=event.localPos()
            position=None
            if area.contains(point) and self.coordinates is not None:
                image_point=self.image_point(point)
                if image_point is not None:position=self.coordinates.millimetres(*image_point)
            text=(f"鼠标：X = {position[0]:.1f} mm，Y = {position[1]:.1f} mm" if position
                  else "鼠标在台面外" if self.coordinates else "未可靠定位内库，暂不可换算坐标")
            self.coordinate_hovered.emit(text)
        super().mouseMoveEvent(event)

    # 鼠标移出画布时清理悬停读数，不改变已保存球桌尺寸或校准。
    def leaveEvent(self,event):
        if self.coordinate_mode:self.coordinate_hovered.emit("鼠标已离开画面")
        super().leaveEvent(event)

    # 将毫米平面中的轴线与球心投影回显示区域；显示精度不等于物理测量精度。
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

    # 只处理有效视频区域内的点击，并把原图坐标通过信号交给主窗口。
    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton and self.selection_enabled
                and self.frame is not None and not self.image.isNull()):
            point = event.localPos()
            target = self.video_rect()
            if target.contains(point):
                image_point=self.image_point(point)
                if image_point is not None:self.element_clicked.emit(self.frame.copy(), *image_point)
        super().mousePressEvent(event)

    # 按原图、几何/袋口、检测/轨迹、坐标和既有标注顺序绘制。
    # 这里不运行模型或更新进球计数，避免重绘频率改变识别语义。
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.image.isNull():
            painter.setPen(Qt.white)
            painter.drawText(self.rect().adjusted(24, 24, -24, -24),
                             Qt.AlignCenter | Qt.TextWordWrap, self.message)
        else:
            area=self.video_rect()
            if self.is_rotated():
                painter.save()
                painter.translate(area.right(),area.top());painter.rotate(90)
                painter.drawImage(QRectF(0,0,area.height(),area.width()),self.image)
                painter.restore()
            else:painter.drawImage(area,self.image)
            self.draw_track_paths(painter)
            occupied = []
            # Draw large table boxes first so small ball boxes stay visible.
            items = sorted(self.detections, key=lambda d: (d["box"][2]-d["box"][0]) *
                           (d["box"][3]-d["box"][1]), reverse=True)
            colors = {"billiards_table": "#40C4FF", "white_ball": "#B2FF59", "black_ball": "#FF9F43"}
            for item in items:
                if self.selected_box is not None and item["box"]==self.selected_box:
                    continue
                if item['name']=='billiards_table' or not self.show_ball_boxes:continue
                self.draw_detection(painter, item["box"], chinese_label(item["name"])+(f" B{item['ball_id']}" if item.get("ball_id") else "")+(" · 已选" if item.get("track_id") in self.marked_ids else ""),
                                    ("#FF9F43" if item.get("track_id") in self.marked_ids else colors.get(item["name"], "#FFEB00")), 14, occupied, item.get("polygon"))
            if self.show_ball_boxes and self.selected_box is not None and self.selected_polygon is None:
                self.draw_detection(painter, self.selected_box, self.selected_label,
                                    "#FFEB00", 20, occupied, self.selected_polygon)
            self.draw_pockets(painter)
            self.draw_coordinates(painter)
            if self.exam_circles and self.calibration_polygon is not None:
                from src.exam_editor import draw_marks
                draw_marks(painter,TableCoordinates(self.table_outline(),self.dimensions),self.exam_circles,self.display_point)
            scope=self.table_outline()
            if scope and (self.show_pocket_overlays or self.pocket_result is None):
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor('#40C4FF'),2))
                painter.drawPolygon(self.canvas_polygon(scope))

    # 绘制历史实测轨迹并尊重缺失区间的断点，避免跨越遮挡画出虚假的连续线。
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

    # 按 observed/manual/estimated/missing 的状态区别绘制入口和文字。
    # 袋口可显示不等于它已被状态机视为可信入袋证据。
    def draw_pockets(self, painter):
        result = self.pocket_result
        if not result or not self.show_pocket_overlays:
            return
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        table = result['table']
        names = {'TOP_LEFT':'左上袋','TOP_MIDDLE':'上中袋','TOP_RIGHT':'右上袋',
                 'BOTTOM_LEFT':'左下袋','BOTTOM_MIDDLE':'下中袋','BOTTOM_RIGHT':'右下袋'}
        # Pocket IDs describe the table plane; labels should describe the screen.
        located=[]
        for p in result['pockets']:
            raw=p['center'] or p['candidate_center']
            point=self.display_point(raw['x'],raw['y']) if raw else None
            located.append((p,{'x':point.x(),'y':point.y()} if point is not None else None))
        if len(located)==6 and all(point is not None for _,point in located):
            corners=np.array([[p.x(),p.y()] for p in self.canvas_polygon(table['playable_polygon'])])
            if np.ptp(corners[:,1])>np.ptp(corners[:,0]):
                ordered=sorted(located,key=lambda item:item[1]['x'])
                groups=[sorted(ordered[:3],key=lambda item:item[1]['y']),
                        sorted(ordered[3:],key=lambda item:item[1]['y'])]
                labels=[['左上袋','左中袋','左下袋'],['右上袋','右中袋','右下袋']]
            else:
                ordered=sorted(located,key=lambda item:item[1]['y'])
                groups=[sorted(ordered[:3],key=lambda item:item[1]['x']),
                        sorted(ordered[3:],key=lambda item:item[1]['x'])]
                labels=[['左上袋','上中袋','右上袋'],['左下袋','下中袋','右下袋']]
            names={p['id']:label for group,words in zip(groups,labels) for (p,_),label in zip(group,words)}
        states = {'observed':('已验证','#52E377'), 'estimated':('估算','#FFDB4D'), 'missing':('未定位','#FF7070'), 'manual':('人工确认','#00E5FF')}
        occupied = []
        font = QFont(self.font());font.setPixelSize(13);font.setBold(True);painter.setFont(font)
        area = self.video_rect()
        for pocket in result['pockets']:
            state, color = states[pocket['status']]
            highlight = QColor(color)
            painter.setPen(QPen(highlight, 2, Qt.SolidLine if pocket['status'] in ('observed','manual') else Qt.DashLine))
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

    # 将原图多边形各顶点映射为 Qt 多边形，复用与图像相同的旋转/缩放规则。
    def canvas_polygon(self, polygon):
        return QPolygonF([self.display_point(x,y) for x,y in polygon])

    # 根据目标类型绘制球框或台面多边形，并布置文字及引线以减少重叠。
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


# 主窗口持有播放、校准、选择、异步请求和事件会话；业务判定交给独立 src 模块。
class MainWindow(QMainWindow):
    identify_requested = pyqtSignal(object, float, float, int)
    identify_all_requested = pyqtSignal(object, int)
    live_requested = pyqtSignal(object, int)
    load_requested = pyqtSignal()
    calibration_requested = pyqtSignal(object, int, object)
    pocket_requested = pyqtSignal(object, int, object)

    # 建立控件、计时器、会话容器及工作线程信号；加载默认配置但不自动开始播放。
    # camera_geometry 保存固定基准副本，工作变换另存，防止反复变换造成累计漂移。
    def __init__(self, video_path=DEFAULT_VIDEO, image_dir=DEFAULT_IMAGES, start_with_images=False):
        super().__init__()
        self.setWindowTitle("台球视觉识别系统")
        self.resize(1200, 800)
        from src.calibration_presets import CalibrationPresets
        self.calibration_presets=CalibrationPresets(ROOT/"config"/"venues")
        self.active_calibration_preset=self.calibration_presets.data["default"]
        self.camera_alignment=None
        self.camera_geometry=None
        self.camera_anchor=np.eye(3)
        self.camera_matrix=np.eye(3)
        self.exam_dialog=None
        self.exam_pending=None
        self.exam_save_timer=QTimer(self)
        self.exam_save_timer.setSingleShot(True)
        self.exam_save_timer.timeout.connect(self.flush_exam_marks)
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
        profile_bar=QWidget();profile_layout=QHBoxLayout(profile_bar)
        profile_layout.setContentsMargins(16,4,16,4)
        profile_layout.addWidget(QLabel("球房／球桌配置："))
        self.calibration_preset_combo=QComboBox();self.calibration_preset_combo.setMinimumSize(220,44)
        self.calibration_preset_combo.setAccessibleName("选择球房和球桌的已保存校准")
        self.calibration_preset_combo.setToolTip("选择同一球桌与固定镜头的配置；球桌范围、袋口和尺寸直接恢复")
        self.add_calibration_preset_button=QPushButton("另存为配置")
        self.add_calibration_preset_button.setToolTip("将当前校准另存为球房／球桌配置；每个新球桌首次校准一次，之后直接选择使用")
        self.default_calibration_preset_button=QPushButton("设为默认")
        for button in (self.add_calibration_preset_button,self.default_calibration_preset_button):button.setMinimumSize(100,44)
        profile_layout.addWidget(self.calibration_preset_combo)
        profile_layout.addWidget(self.add_calibration_preset_button)
        profile_layout.addWidget(self.default_calibration_preset_button)
        self.exam_button=QPushButton("考题")
        self.exam_button.setMinimumSize(100,44)
        self.exam_button.clicked.connect(self.open_exam)
        profile_layout.addWidget(self.exam_button)
        self.calibration_state_label=QLabel("校准：未保存")
        self.calibration_state_label.setWordWrap(True)
        profile_layout.addWidget(self.calibration_state_label,1)
        profile_layout.addStretch()
        self.refresh_calibration_presets()
        self.calibration_preset_combo.currentIndexChanged.connect(self.select_calibration_preset)
        self.add_calibration_preset_button.clicked.connect(self.add_calibration_preset)
        self.default_calibration_preset_button.clicked.connect(self.set_default_calibration_preset)
        layout.addWidget(profile_bar)
        controls = QWidget()
        buttons = QHBoxLayout(controls)
        buttons.setContentsMargins(16, 8, 16, 8)
        buttons.setSpacing(12)
        self.play_button = QPushButton("播放")
        self.yolo_button = QPushButton("YOLO：关闭")
        self.yolo_button.setCheckable(True)
        self.yolo_button.setMinimumSize(150, 44)
        self.yolo_button.setToolTip("开启后点选球，再播放只跟踪该球；点击识别全部才识别所有目标")
        buttons.addStretch()
        for button in (self.play_button,):
            button.setMinimumSize(100, 44)
            button.setEnabled(False)
            buttons.addWidget(button)
        buttons.addWidget(self.yolo_button)
        self.identify_all_button = QPushButton("全部识别")
        self.identify_all_button.setMinimumSize(120, 44)
        self.identify_all_button.setEnabled(False)
        self.identify_all_button.setToolTip("点击识别全部目标，再次点击取消；播放时持续更新")
        buttons.addWidget(self.identify_all_button)
        self.pocket_button = QPushButton("自动检测袋口")
        self.pocket_button.setMinimumSize(150, 44)
        self.pocket_button.setEnabled(False)
        self.pocket_button.setToolTip("重新读取当前画面，自动校准球桌范围与六个袋口；不识别球体，可重复检测")
        buttons.addWidget(self.pocket_button)
        self.pocket_visibility_button=QPushButton("隐藏袋口识别")
        self.pocket_visibility_button.setCheckable(True)
        self.pocket_visibility_button.setMinimumSize(130,44)
        self.pocket_visibility_button.hide()
        self.pocket_visibility_button.toggled.connect(self.toggle_pocket_visibility)
        self.canvas.pockets_available.connect(self.pocket_visibility_button.setVisible)
        buttons.addWidget(self.pocket_visibility_button)
        self.pot_button=QPushButton("进球检测")
        self.pot_button.setCheckable(True);self.pot_button.setMinimumSize(110,44)
        self.pot_button.setToolTip("自动观察黑白球接近袋口、消失及静止后的复核；证据不足会提示人工复核")
        self.pot_button.toggled.connect(self.toggle_pot_detection)
        buttons.addWidget(self.pot_button)
        self.pot_count_label=QLabel("（0）")
        self.pot_count_label.setToolTip("当前视频已确认的所有球入袋总数（含白球）；疑似与待复核不计入")
        buttons.addWidget(self.pot_count_label)
        self.pot_counts={};self.pot_count_epoch=0;self.pot_reset_count_on_play=False
        buttons.addStretch()
        layout.addWidget(controls)
        coordinate_bar=QWidget()
        coordinate_layout=QHBoxLayout(coordinate_bar)
        coordinate_layout.setContentsMargins(16,0,16,8)
        self.coordinate_button=QPushButton("坐标模式")
        self.coordinate_button.setCheckable(True);self.coordinate_button.setMinimumSize(110,44)
        self.coordinate_button.setToolTip("以当前识别台面四角建立毫米坐标；开启时暂停视频，移动鼠标查看坐标，点击球查看球心坐标")
        self.dimensions_button=QPushButton("球桌尺寸")
        self.dimensions_button.setMinimumSize(100,44)
        self.trails_button=QComboBox()
        self.trails_button.addItems(["全部运动球轨迹","白球运动轨迹","其他球运动轨迹","当前所选球运动轨迹","关闭轨迹"])
        self.trails_button.setCurrentIndex(4)
        self.trails_button.currentIndexChanged.connect(self.toggle_trails)
        self.trails_button.setToolTip("前三种模式播放时自动跟踪，无需选球；所选球模式需先点选目标；关闭轨迹隐藏所有轨迹")
        coordinate_layout.addWidget(self.trails_button)
        self.coordinate_readout=QLabel("坐标模式未开启")
        self.ball_coordinate_readout=QLabel("球心：未选择")
        self.ball_coordinate_readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        coordinate_layout.addWidget(self.coordinate_button)
        coordinate_layout.addWidget(self.dimensions_button)
        self.manual_calibration_button=QPushButton("手动校准")
        self.manual_calibration_button.setMinimumSize(100,44)
        self.manual_calibration_button.setToolTip("点选台呢内沿四角及六袋口，按当前来源保存，更新球桌与袋口")
        self.manual_calibration_button.clicked.connect(self.edit_calibration)
        coordinate_layout.addWidget(self.manual_calibration_button)
        coordinate_layout.addWidget(self.coordinate_readout,1)
        coordinate_layout.addWidget(self.ball_coordinate_readout)
        layout.addWidget(coordinate_bar)
        self.pot_readout=QLabel("进球检测未开启")
        self.pot_readout.setWordWrap(True);self.pot_readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.pot_readout.setContentsMargins(16,0,16,8)
        layout.addWidget(self.pot_readout)
        self.pot_adapter=None;self.pot_recording=None;self.pot_last_snapshot=None
        self.pot_observer=None
        self.dimension_path=(self.calibration_presets.path(self.active_calibration_preset).with_suffix('.dimensions.json') if self.active_calibration_preset else ROOT/'config'/'table_dimensions.json')
        try:
            self.canvas.dimensions=TableDimensions(**json.loads(self.dimension_path.read_text())) if self.dimension_path.exists() else TableDimensions()
        except (ValueError,TypeError,OSError):
            self.canvas.dimensions=TableDimensions()
        self.refresh_dimension_label()
        self.coordinate_button.toggled.connect(self.toggle_coordinates)
        self.dimensions_button.clicked.connect(self.configure_dimensions)
        self.canvas.coordinate_hovered.connect(self.coordinate_readout.setText)
        self.setCentralWidget(content)
        self.video_fullscreen=False
        self.fullscreen_controls=[gallery,profile_bar,controls,coordinate_bar,self.pot_readout,self.statusBar()]
        self.canvas.fullscreen_requested.connect(self.toggle_video_fullscreen)
        self.fullscreen_escape=QShortcut(QKeySequence(Qt.Key_Escape),self)
        self.fullscreen_escape.activated.connect(lambda:self.toggle_video_fullscreen() if self.video_fullscreen else None)
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
        self.selection_busy=False;self.resume_after_selection=False
        self.detection_mode="idle"
        self.single_table_polygon=None
        self.live_last_result_time=None
        self.media_generation = 0
        self.pocket_busy = False
        self.pocket_thread = None
        self.pocket_worker = None
        self.last_pocket_result = None
        self.calibration_busy=False
        self.calibration_thread=None
        self.calibration_worker=None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.play_button.clicked.connect(self.toggle_playback)
        self.yolo_button.toggled.connect(self.toggle_yolo)
        self.identify_all_button.clicked.connect(self.identify_all_elements)
        self.pocket_button.clicked.connect(self.detect_pockets)
        self.canvas.element_clicked.connect(self.identify_element)
        self.video_path = Path(video_path).expanduser().resolve()
        self.video_paths=sorted((p.resolve() for p in self.video_path.parent.iterdir() if p.is_file() and p.suffix.lower() in {'.mp4','.mov','.mkv','.avi','.m4v'}),key=lambda p:p.name.casefold()) if self.video_path.parent.is_dir() else []
        if self.video_path not in self.video_paths:self.video_paths.append(self.video_path)
        self.video_index=self.video_paths.index(self.video_path)
        self.source_combo.currentIndexChanged.connect(self.switch_source)
        self.previous_button.clicked.connect(lambda: self.navigate_media(-1))
        self.next_button.clicked.connect(lambda: self.navigate_media(1))
        self.previous_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.next_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.previous_shortcut.activated.connect(self.previous_button.click)
        self.next_shortcut.activated.connect(self.next_button.click)
        if start_with_images:
            self.source_combo.setCurrentIndex(1)
        else:
            self.switch_source(0)
        if self.calibration_presets.error:
            self.statusBar().showMessage("球桌配置列表无法读取，原文件已保留："+self.calibration_presets.error)

    # 只旋转并适配当前视频区域，保留下方操作栏；没有调用系统全屏。
    def toggle_video_fullscreen(self):
        self.video_fullscreen=not self.video_fullscreen
        self.canvas.rotate_portrait=self.video_fullscreen
        self.canvas.fullscreen_button.setText("恢复竖屏" if self.video_fullscreen else "旋转放大")
        self.canvas.fullscreen_button.adjustSize()
        self.canvas.fullscreen_button.move(max(0,self.canvas.width()-self.canvas.fullscreen_button.width()-12),12)
        self.canvas.update()

    # 以会话 epoch、shot_id、球身份为键覆盖判定；同事件重复返回不会重复加分。
    # NOT_POTTED 修订会撤销该键先前的确认，因此不能简单对每个 POTTED 做累加。
    def update_pot_count(self,results=(),clear=False):
        if clear:self.pot_counts.clear()
        for result in results:
            for event in result['events']:
                key=(self.pot_count_epoch,result['shot_id'],event['ball'])
                self.pot_counts[key]=event['event']=='POTTED'
        self.pot_count_label.setText(f"（{sum(self.pot_counts.values())}）")

    # 切换会话前把未完成杆以指定原因收尾为 UNKNOWN，关闭记录器并递增计数世代。
    def reset_pot_session(self,reason):
        if self.pot_adapter is not None and not self.pot_adapter.engine.completed and self.pot_adapter.engine.timestamp is not None:
            results=self.pot_adapter.engine.finish(force_unknown=reason)
            self.update_pot_count(results)
            if self.pot_recording:self.pot_recording.write(self.pot_adapter.engine.snapshot(),results)
        if self.pot_recording:self.pot_recording.close()
        self.pot_observer=None
        self.pot_count_epoch+=1
        self.pot_adapter=None;self.pot_recording=None;self.pot_last_snapshot=None
        self.pot_readout.setText("进球检测："+reason if self.pot_button.isChecked() else "进球检测未开启")
        self.pot_resume_after_calibration=False

    # 开关进球观察并使旧 YOLO 请求失效；无袋口几何时先启动独立几何检测。
    def toggle_pot_detection(self,enabled):
        self.yolo_generation+=1
        self.reset_pot_session("等待当前视频画面")
        self.pot_button.setText("关闭进球检测" if enabled else "进球检测")
        if not enabled:return
        if self.media_mode!="video":
            self.pot_readout.setText("进球检测需要完整视频，请切换到视频后播放")
            return
        self.pocket_visibility_button.setChecked(False)
        if self.canvas.pocket_result is None:
            self.pot_readout.setText("进球检测：正在定位球桌与袋口…")
            resume=self.timer.isActive()
            self.detect_pockets()
            self.pot_resume_after_calibration=resume
            return
        self.prepare_pot_detection()

    # 用已有可信几何建立像素观察器、事件适配器和本地记录器。
    # YOLO 只在轨迹/选择需要时开启；进球像素观察本身不等待模型结果。
    def prepare_pot_detection(self):
        if not self.pot_button.isChecked() or self.media_mode!="video":return
        if self.pot_adapter is None:
            if self.canvas.pocket_result is None:
                self.pot_readout.setText("进球检测：请先自动检测袋口，再播放完整击球视频")
                return
            try:
                from src.vision.pocket_event.adapter import PocketEventAdapter
                from src.vision.pocket_event.recording import EventRecording
                self.pot_adapter=PocketEventAdapter(self.canvas.pocket_result,self.canvas.dimensions)
                from src.vision.pocket_event.pixel_observer import PixelBallObserver
                self.pot_observer=PixelBallObserver(self.canvas.pocket_result['table']['playable_polygon'])
                self.pot_recording=EventRecording(ROOT/'data'/'pocket_events',self.video_path)
            except (ValueError,KeyError,cv2.error) as exc:
                self.pot_readout.setText(f"进球检测暂不可用：{exc}");return
            self.pot_readout.setText("黑球：等待完整轨迹；白球：等待完整轨迹 · 播放视频开始观察")
        if (self.automatic_trails() or self.detection_mode!="idle") and not self.yolo_button.isChecked():self.yolo_button.setChecked(True)
        self.maybe_submit_live()

    # 每个实际解码帧运行像素观察；时间取视频帧号/FPS，而非模型耗时或墙钟。
    def observe_current_pot_frame(self,frame):
        if not self.pot_button.isChecked():return
        if self.pot_observer is None or self.pot_adapter is None:self.prepare_pot_detection()
        if self.pot_observer is None or self.pot_adapter is None:return
        fps=self.capture.get(cv2.CAP_PROP_FPS)
        index=max(0,self.capture.get(cv2.CAP_PROP_POS_FRAMES)-1)
        if fps>0:
            self.observe_pot_events(index/fps,self.pot_observer.observe(frame),frame)

    # 将实测检测送入适配器，应用可修订计数并更新状态文字。
    # 仅状态签名变化或新结果时写事件关键帧，避免每次 UI 刷新都存图。
    def observe_pot_events(self,timestamp,detections,frame):
        if self.pot_adapter is None:
            self.prepare_pot_detection()
            if self.pot_adapter is None:return
        try:
            results=self.pot_adapter.update(timestamp,detections,frame)
            self.update_pot_count(results)
            snapshot=self.pot_adapter.engine.snapshot()
            labels={'TRACKING':'桌面可见','APPROACHING_POCKET':'接近袋口','POT_CANDIDATE':'疑似进袋，等待复核',
                    'POTTED':'已进袋','NOT_POTTED':'未进袋','UNKNOWN':'无法确认，需要人工复核'}
            names={'TOP_LEFT':'左上袋','TOP_MIDDLE':'上中袋','TOP_RIGHT':'右上袋',
                   'BOTTOM_LEFT':'左下袋','BOTTOM_MIDDLE':'下中袋','BOTTOM_RIGHT':'右下袋'}
            parts=[]
            for ball,title in (('black_ball','黑球'),('white_ball','白球')):
                state=snapshot['balls'][ball]
                suffix=(' · '+names.get(state['candidate_pocket'],state['candidate_pocket'])) if state['candidate_pocket'] else ''
                parts.append(title+'：'+labels[state['state']]+suffix)
            signature=tuple((ball,state['state'],state['candidate_pocket']) for ball,state in snapshot['balls'].items())
            targets=[state for ball,state in snapshot['balls'].items() if ball.startswith('billiard_ball:')]
            if targets:
                pending=sum(state['state'] in ('UNKNOWN','POT_CANDIDATE') and state['candidate_pocket'] is not None for state in targets)
                parts.append(f"进球候选：待复核 {pending} 项")
            if signature!=self.pot_last_snapshot or results:
                self.pot_recording.write(snapshot,results,frame)
                self.pot_last_snapshot=signature
            self.pot_readout.setText(' | '.join(parts)+f" · 已确认 {sum(self.pot_counts.values())} 球 · 实时观察 {timestamp:.2f}s"
                +(f" · 记录失败：{self.pot_recording.error}" if self.pot_recording.error else ''))
            self.pot_readout.setToolTip('判定记录：'+str(self.pot_recording.path))
        except (ValueError,KeyError,cv2.error) as exc:
            self.pot_readout.setText(f"进球判定证据异常，需人工复核：{exc}")

    # 前三种轨迹范围需要自动观测整桌；所选球和关闭轨迹走不同显示/提交规则。
    def automatic_trails(self):
        return self.trails_button.currentIndex()<3

    # 只改变袋口叠加层可见性，不删除校准或停掉进球分析。
    def toggle_pocket_visibility(self, hidden):
        self.canvas.show_pocket_overlays=not hidden
        self.pocket_visibility_button.setText("显示袋口识别" if hidden else "隐藏袋口识别")
        self.canvas.update()

    # 切换观察范围时递增请求世代并清理过时选择，必要时加载模型提交整桌请求。
    def toggle_trails(self, mode):
        self.canvas.show_trails=mode!=4
        self.yolo_generation+=1  # Results from a former selected-only request are obsolete.
        self.pending_selection=None
        if mode<3:
            self.selected_targets.clear();self.live_selected_id=None
            self.canvas.selected_box=None;self.canvas.selected_polygon=None;self.canvas.marked_ids.clear()
            if self.detection_mode=="single":self.detection_mode="idle"
            self.ball_coordinate_readout.setText("球心：未选择")
            if not self.yolo_button.isChecked():self.yolo_button.setChecked(True)
            self.maybe_submit_live()
        self.canvas.show_ball_boxes=self.detection_mode=="all" or self.detection_mode=="single" and mode>=3
        if mode==3:
            if self.selected_targets:
                if not self.yolo_button.isChecked():self.yolo_button.setChecked(True)
                self.maybe_submit_live()
            else:self.statusBar().showMessage("所选球轨迹：请先点击视频中的球，再播放")
        self.sync_tracks();self.canvas.update();self.update_all_button()

    # 按全部/白球/其他球/所选球筛选显示轨迹，同时将身份位置流式写到本地记录。
    # lost 状态不记录为真实球心；写盘失败只提示，不停止画面显示。
    def sync_tracks(self):
        identities=self.live_tracking.identities
        visible=set(self.selected_targets) if self.detection_mode=="single" and not self.automatic_trails() else set(identities.states)
        mode=self.trails_button.currentIndex()
        selected=set(self.selected_targets)
        if self.live_selected_id is not None:selected.add(self.live_selected_id)
        self.canvas.track_paths={key:list(state['path']) for key,state in identities.states.items()
            if key in visible and (mode==0 or mode==1 and state['name']=='white_ball'
                                   or mode==2 and state['name']!='white_ball' or mode==3 and key in selected)}
        if self.media_mode!="video" or (self.detection_mode=="idle" and not self.automatic_trails()) or self.capture is None or not self.playback_ready:return
        table=next((d for d in self.live_tracking.detections if d['name']=='billiards_table'),None)
        polygon=self.canvas.table_outline()
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

    # 更新参考尺寸按钮文字与提示，区分参考尺寸和用户测量尺寸。
    def refresh_dimension_label(self):
        d=self.canvas.dimensions
        prefix="参考" if d.reference else "配置"
        self.dimensions_button.setText(f"{prefix}尺寸：{d.x_mm:g} × {d.y_mm:g} mm")
        self.dimensions_button.setToolTip("内库之间的比赛区域尺寸；X 为上边方向，Y 为左边向下。点击修改并保存。")

    # 校验并保存用户输入的 X/Y 毫米尺寸，然后刷新依赖该尺寸的坐标映射。
    def configure_dimensions(self):
        self.close_exam()
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
        self.reset_pot_session("球桌尺寸已改变")
        self.refresh_dimension_label()
        self.ball_coordinate_readout.setText("球心：未选择")
        if self.canvas.coordinate_mode:self.canvas.calibrate_coordinates()
        self.canvas.update()

    # 打开已有截图编辑器并绑定当前几何；本轮只解释现有代码，不新增考题功能。
    def open_exam(self):
        if self.canvas.frame is None or self.canvas.calibration_polygon is None:
            self.statusBar().showMessage("请先自动或手动校准球桌，再打开考题")
            return
        if self.exam_dialog is not None:
            self.exam_dialog.show();self.exam_dialog.raise_();self.exam_dialog.activateWindow();return
        from src.exam_editor import ExamDialog
        self.exam_dialog=ExamDialog(self.canvas.frame.copy(),self.canvas.table_outline(),self.canvas.dimensions,self.canvas.exam_circles,self)
        self.exam_dialog.changed.connect(self.change_exam_marks)
        self.exam_dialog.finished.connect(self.finish_exam)
        self.exam_dialog.show()

    # 对话框结束时刷新已有标注并释放对话框引用。
    def finish_exam(self,*args):
        dialog=self.exam_dialog;self.exam_dialog=None
        if dialog is not None:dialog.deleteLater()
        self.flush_exam_marks()

    # 接收标注副本，刷新视频叠加并通过单次计时器合并连续拖动产生的保存请求。
    def change_exam_marks(self,marks):
        self.canvas.exam_circles=deepcopy(marks);self.canvas.update()
        self.exam_pending=(self.calibration_profile_path().with_suffix('.exam.json'),deepcopy(marks))
        self.exam_save_timer.start(250)

    # 将最后一次编辑结果写到对应配置；保存对象是标注，不是重新检测的台面。
    def flush_exam_marks(self):
        self.exam_save_timer.stop()
        if self.exam_pending is None:return
        path,marks=self.exam_pending
        try:
            from src.calibration_state import write_json
            write_json(path,{'version':1,'circles':marks})
            self.exam_pending=None
        except OSError as exc:self.statusBar().showMessage(f"考题标注保存失败：{exc}")

    # 从当前球桌配置加载并校验标注；无效或缺失记录不伪造圆圈。
    def load_exam_marks(self):
        self.canvas.exam_circles=[]
        path=self.calibration_profile_path().with_suffix('.exam.json')
        try:
            if path.exists():
                from src.exam_editor import validate_marks
                data=json.loads(path.read_text())
                if data.get('version')!=1:raise ValueError('不支持的考题版本')
                self.canvas.exam_circles=validate_marks(data['circles'])
        except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
            self.statusBar().showMessage(f"考题标注读取失败：{exc}")
        self.canvas.update()

    # 先关闭编辑器并处理已有保存流程，避免换源后旧编辑继续写入新配置。
    def close_exam(self):
        if self.exam_dialog is not None:self.exam_dialog.close()
        self.flush_exam_marks()

    # 集中同步校准相关控件的可用性，防止异步校准期间重复提交。
    def calibration_controls(self, busy):
        self.calibration_busy=busy
        self.pocket_button.setText("球桌与袋口检测中…" if busy else "自动检测袋口")
        for widget in (self.play_button,self.previous_button,self.next_button,self.yolo_button,
                       self.identify_all_button,self.pocket_button,self.coordinate_button,self.dimensions_button,self.source_combo,self.manual_calibration_button,self.pot_button,self.calibration_preset_combo,self.add_calibration_preset_button,self.default_calibration_preset_button,self.exam_button):
            widget.setEnabled(not busy)
        self.previous_shortcut.setEnabled(not busy)
        self.next_shortcut.setEnabled(not busy)
        self.canvas.selection_enabled=not busy and self.yolo_button.isChecked()
        if not busy:
            self.refresh_calibration_presets()
            self.update_navigation()
            self.update_all_button()

    # 重建球房/球桌配置下拉框，并保留当前有效配置的选择。
    def refresh_calibration_presets(self):
        self.calibration_preset_combo.blockSignals(True)
        self.calibration_preset_combo.clear()
        default=self.calibration_presets.data['default']
        self.calibration_preset_combo.addItem("按视频／图片分别保存"+("（默认）" if default is None else ""),None)
        for profile in self.calibration_presets.data['profiles']:
            self.calibration_preset_combo.addItem(profile['name']+("（默认）" if profile['id']==default else ""),profile['id'])
        self.calibration_preset_combo.setCurrentIndex(max(0,self.calibration_preset_combo.findData(self.active_calibration_preset)))
        self.calibration_preset_combo.blockSignals(False)
        self.default_calibration_preset_button.setEnabled(self.active_calibration_preset!=default and not self.calibration_presets.error)
        self.add_calibration_preset_button.setEnabled(not self.calibration_presets.error)

    # 询问用户配置名称，合法输入才进入创建流程。
    def add_calibration_preset(self):
        name,accepted=QInputDialog.getText(self,"新增球房／球桌配置","配置名称（例如：朝阳俱乐部 · 1号桌）")
        if not accepted:return
        try:self.create_calibration_preset(name)
        except (OSError,ValueError) as exc:self.statusBar().showMessage(f"配置保存失败：{exc}")

    # 包装创建流程和错误提示，让创建失败保留原有配置。
    def create_calibration_preset(self,name):
        previous=deepcopy(self.calibration_presets.data)
        try:return self._create_calibration_preset(name)
        except (OSError,ValueError):
            if self.calibration_presets.data!=previous:
                self.calibration_presets.data=previous
                self.calibration_presets.save()
            raise

    # 建立命名配置并复制当前可用校准/标注，避免仅创建一个空名字。
    def _create_calibration_preset(self,name):
        self.close_exam()
        from src.calibration_state import save_automatic,write_json
        old_path=self.calibration_profile_path()
        manual=json.loads(old_path.read_text()) if old_path.exists() else None
        identity=self.calibration_presets.create(name)
        path=self.calibration_presets.path(identity)
        if manual is not None:write_json(path,manual)
        if self.canvas.frame is not None and self.canvas.calibration_polygon is not None and self.canvas.pocket_result is not None:
            pockets=deepcopy(self.canvas.pocket_result)
            pockets['table']['scope_polygon']=self.canvas.recognition_polygon or self.canvas.calibration_polygon
            save_automatic(path.with_suffix('.auto.json'),self.canvas.frame,
                           {'polygon':self.canvas.calibration_polygon,'pockets':pockets,'calibration_report':{'decision':'saved_preset'}},manual)
        write_json(path.with_suffix('.exam.json'),{'version':1,'circles':self.canvas.exam_circles})
        write_json(path.with_suffix('.dimensions.json'),self.canvas.dimensions.__dict__)
        self.active_calibration_preset=identity
        self.refresh_calibration_presets()
        self.select_calibration_preset(self.calibration_preset_combo.currentIndex())
        return identity

    # 显式保存启动默认配置；当前下拉选择与启动默认是两个不同状态。
    def set_default_calibration_preset(self):
        try:self.calibration_presets.set_default(self.active_calibration_preset)
        except (OSError,ValueError) as exc:
            self.statusBar().showMessage(f"默认配置保存失败：{exc}");return
        self.refresh_calibration_presets()
        self.statusBar().showMessage("已设为默认，下次打开直接使用该配置；后续校准会自动保存")

    # 先清理旧会话再恢复所选配置，避免不同球房的坐标/事件相互混用。
    def select_calibration_preset(self,index):
        if self.calibration_busy:return
        self.pause_video()
        self.active_calibration_preset=self.calibration_preset_combo.itemData(index)
        self.invalidate_results()
        self.dimension_path=(self.calibration_presets.path(self.active_calibration_preset).with_suffix('.dimensions.json') if self.active_calibration_preset else ROOT/'config'/'table_dimensions.json')
        try:self.canvas.dimensions=TableDimensions(**json.loads(self.dimension_path.read_text())) if self.dimension_path.exists() else TableDimensions()
        except (OSError,ValueError,TypeError):self.canvas.dimensions=TableDimensions()
        self.refresh_dimension_label()
        restored=self.restore_saved_calibration()
        self.refresh_calibration_presets()
        name=self.calibration_preset_combo.currentText()
        self.statusBar().showMessage(f"已加载 {name}，点击播放即可使用" if restored else f"{name} 尚无适用校准，请先自动或手动校准一次")

    # 命名配置使用其专属目录；未选配置时以来源路径哈希区分视频/图片。
    def calibration_profile_path(self):
        if self.active_calibration_preset:return self.calibration_presets.path(self.active_calibration_preset)
        source=(self.image_paths[self.image_index] if self.media_mode=='images' else self.video_path)
        key=hashlib.sha256(str(source.resolve()).encode('utf-8')).hexdigest()
        return ROOT/'config'/'calibration'/f'{key}.json'

    # 复制固定几何并以参考画面建立配准器；旧版无参考记录仅在当前会话补锚点。
    # 工作变换与磁盘基准分离，这个步骤不把局部遮挡下的新轮廓自动写回配置。
    def start_camera_alignment(self,reference=None,legacy=False):
        self.camera_alignment=None;self.camera_geometry=None
        self.camera_anchor=np.eye(3);self.camera_matrix=np.eye(3)
        if self.media_mode!='video' or self.canvas.calibration_polygon is None:return
        from src.camera_alignment import CameraAlignment
        polygon=deepcopy(self.canvas.calibration_polygon)
        scope=deepcopy(self.canvas.recognition_polygon or polygon)
        self.camera_geometry=(polygon,scope,deepcopy(self.canvas.pocket_result))
        anchor_scope=scope
        if legacy and reference is None:
            # Older records have no reference image. Locate the visible cloth
            # once for this playback session, keeping the disk baseline intact.
            from src.cloth_region import cloth_region
            region=cloth_region(self.canvas.frame,polygon)
            if region and region['color_fill']>.9 and region['connected_color_coverage']>.95:
                candidate=np.asarray(region['polygon'],float)
                prior=np.asarray(scope,float)
                shift=np.max(np.linalg.norm(candidate-prior,axis=1))
                ratio=abs(cv2.contourArea(candidate.astype(np.float32))/cv2.contourArea(prior.astype(np.float32)))
                if shift<min(self.canvas.frame.shape[:2])*.12 and .75<ratio<1.3:
                    self.camera_anchor=cv2.getPerspectiveTransform(np.float32(scope),np.float32(candidate))
                    anchor_scope=candidate.tolist()
        try:
            self.camera_alignment=CameraAlignment(self.canvas.frame,anchor_scope,reference)
            self.update_camera_alignment(self.canvas.frame,force=True)
        except (ValueError,cv2.error) as exc:
            self.camera_alignment=None
            self.statusBar().showMessage(f'画面对齐暂不可用，保留已保存基准：{exc}')

    # 把相对于参考图的变换应用到基准副本，并同步适配器、画布与模型范围。
    # 失败保留显示几何；当前实现尚未把 lost 接入统一进球许可，不能据注释当作已安全保护。
    def update_camera_alignment(self,frame,force=False):
        if self.camera_alignment is None or self.camera_geometry is None:return
        from src.camera_alignment import transform_geometry
        result=self.camera_alignment.update(frame,force)
        note='对齐证据不足，暂保留位置' if self.camera_alignment.lost else '已随画面对齐'
        self.calibration_state_label.setText(getattr(self,'calibration_state_text','固定基准')+' · '+note)
        matrix=self.camera_alignment.matrix@self.camera_anchor
        if result is None and not force:return
        if np.allclose(matrix,self.camera_matrix,atol=1e-5):return
        delta=matrix@np.linalg.inv(self.camera_matrix)
        polygon,scope,pockets=transform_geometry(*self.camera_geometry,matrix)
        self.camera_matrix=matrix
        self.canvas.calibration_polygon=polygon;self.canvas.recognition_polygon=scope
        self.single_table_polygon=polygon
        self.canvas.pocket_result=pockets;self.last_pocket_result=pockets
        if self.yolo_worker is not None:
            self.yolo_worker.calibration_polygon=polygon;self.yolo_worker.scope_polygon=scope
        if self.pot_adapter is not None and pockets is not None:
            self.pot_adapter.align_geometry(pockets,delta)
            if self.pot_observer is not None:
                self.pot_observer.polygon=np.asarray(polygon,np.float32)
                self.pot_observer.cached_shape=None
        self.canvas.calibrate_coordinates()
        self.canvas.update()

    # 读取保存时间/版本与当前对齐状态，用于显示基准来源和可信性提示。
    def refresh_calibration_state(self):
        path=self.calibration_profile_path().with_suffix('.auto.json')
        try:
            data=json.loads(path.read_text())
            when=data.get('saved_at','旧版记录（无时间）')
            self.calibration_state_text=f'固定基准 · {when}'
            self.calibration_state_label.setText(self.calibration_state_text)
            self.calibration_state_label.setToolTip(f'已保存至 {path}\n播放、切换横竖屏、重启均沿用基准；镜头变化时只对齐显示，点击自动检测或手动校准才修改保存值。')
        except (OSError,ValueError,AttributeError):self.calibration_state_label.setText('校准：尚无已保存记录')

    # 优先恢复已保存的自动/人工记录，按帧尺寸换算几何并启动参考配准。
    # 恢复失败不意味着可以把当前遮挡画面保存成新的默认校准。
    def restore_saved_calibration(self):
        """Restore source geometry before any new inference can use the frame."""
        self._calibration_restore_note=None
        if self.canvas.frame is None:return False
        try:
            path=self.calibration_profile_path()
            profile=json.loads(path.read_text()) if path.exists() else None
            from src.calibration_state import load_automatic,profile_digest
            auto_path=path.with_suffix('.auto.json')
            automatic=load_automatic(auto_path,self.canvas.frame,profile,verify_scene=False)
            try:
                prior=json.loads(auto_path.read_text()) if auto_path.exists() else {}
                current_auto=prior.get('manual_digest')==profile_digest(profile)
            except (OSError,ValueError,AttributeError):current_auto=False
            if automatic is None and current_auto:
                self._calibration_restore_note='当前画面宽高比或校准版本与保存记录不匹配，未套用旧范围，请检查来源或重新校准'
                self.statusBar().showMessage(self._calibration_restore_note)
                return False
            if automatic is not None:
                polygon,pockets=automatic
                # Migrate legacy scopes once, then restore the saved result.
                if not prior.get('scope_frozen'):
                    from src.calibration_state import save_automatic
                    save_automatic(auto_path,self.canvas.frame,{'polygon':polygon,'pockets':pockets,'calibration_report':prior.get('report')},profile)
            elif profile is not None:
                from src.manual_calibration import manual_geometry
                polygon,pockets=manual_geometry(self.canvas.frame,profile)
                from src.calibration_state import save_automatic
                save_automatic(auto_path,self.canvas.frame,{'polygon':polygon,'pockets':pockets,'calibration_report':{'decision':'manual_restore'}},profile)
            else:return False
        except (OSError,ValueError,TypeError,KeyError,AttributeError,cv2.error) as exc:
            self.statusBar().showMessage(f"已保存的校准无法恢复，请重新校准：{exc}")
            return False
        self.canvas.calibration_polygon=polygon
        self.canvas.recognition_polygon=pockets['table'].get('scope_polygon') or polygon
        self.single_table_polygon=polygon
        if self.yolo_worker is not None:
            self.yolo_worker.calibration_polygon=polygon
            self.yolo_worker.scope_polygon=self.canvas.recognition_polygon
        self.canvas.calibrate_coordinates()
        self.last_pocket_result=pockets
        self.canvas.set_pocket_result(pockets)
        self.load_exam_marks()
        self.refresh_calibration_state()
        self.start_camera_alignment(prior.get('alignment_reference'),legacy=True)
        saved_at=prior.get('saved_at','此前保存（旧版记录无时间）')
        self._calibration_restore_note=f'已加载固定校准 · 保存于 {saved_at}；保存基准不变，标注随画面移动缩放'
        self.statusBar().showMessage(self._calibration_restore_note)
        self.pocket_button.setToolTip(self._calibration_restore_note+'；点击此按钮会主动重新校准')
        self.canvas.update()
        return True

    # 暂停后打开人工角点/袋口编辑器；仅用户接受并通过校验时保存结果。
    def edit_calibration(self):
        if self.calibration_busy or self.canvas.frame is None:return
        self.pause_video()
        from src.calibration_editor import CalibrationEditor
        dialog=CalibrationEditor(self.canvas.frame.copy(),self)
        if dialog.exec_()!=QDialog.Accepted:return
        path=self.calibration_profile_path()
        try:
            path.with_suffix('.auto.json').unlink(missing_ok=True)
            path.with_suffix('.feedback.json').unlink(missing_ok=True)
            if dialog.clear_saved:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(dialog.profile(),ensure_ascii=False,indent=2))
                temporary.replace(path)
        except OSError as exc:
            self.statusBar().showMessage(f"手动校准保存失败：{exc}")
            return
        self.calibrate_table(collect_sample=not dialog.clear_saved)

    # 显式校准入口：保存纠错上下文、冻结当前任务来源并启动后台搜索。
    def calibrate_table(self, collect_sample=False):
        if self.calibration_busy:return
        if self.canvas.frame is None:
            self.statusBar().showMessage("尚无画面，无法校准")
            return
        self.pause_video()
        self.pocket_visibility_button.setChecked(False)
        source=({'kind':'image','path':str(self.image_paths[self.image_index])}
                if self.media_mode=='images' else
                {'kind':'video','path':str(self.video_path),
                 'frame_index':max(0,int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))-1)})
        try:
            profile_path=self.calibration_profile_path()
            if profile_path.exists():source['manual_profile']=json.loads(profile_path.read_text())
            if collect_sample:source['collect_training_sample']=True
        except (OSError,ValueError) as exc:
            self.statusBar().showMessage(f"已保存的校准无法读取，请用手动校准重新保存：{exc}")
            return
        frame=self.canvas.frame.copy()
        self._calibration_backup={key:deepcopy(getattr(self.canvas,key)) for key in
                                  ('detections','selected_box','selected_polygon','selected_label','pocket_result','calibration_polygon','recognition_polygon','track_paths','marked_ids')}
        self._calibration_backup['frame']=frame
        self._calibration_backup['_main']={key:deepcopy(getattr(self,key)) for key in
                                           ('detection_mode','selected_targets','live_selected_id','live_tracking')}
        if not collect_sample:
            previous=self.canvas.calibration_polygon
            if previous is None:
                table=next((item for item in list(self.selected_targets.values())+self.canvas.detections
                            if item['name']=='billiards_table'),None)
                if table:
                    previous=table.get('polygon')
                    if previous is None:
                        x1,y1,x2,y2=table['box'];previous=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            try:
                from src.calibration_state import begin_feedback
                source['correction_context']=begin_feedback(profile_path.with_suffix('.feedback.json'),frame,previous,source.get('manual_profile'))
                source['correction_context']['previous_scope']=self.canvas.recognition_polygon or previous
                source['correction_context']['previous_pockets']=deepcopy(self.canvas.pocket_result)
            except (OSError,ValueError,cv2.error) as exc:
                self.statusBar().showMessage(f"无法保存纠错记录：{exc}；当前结果保留")
                return
        self.invalidate_results()
        self.live_inflight=None
        self.calibration_controls(True)
        self.statusBar().showMessage("1/4 正在重新读取画面，自动校准球桌与袋口…")
        if self.calibration_thread is None:
            self.calibration_thread=QThread(self)
            self.calibration_worker=CalibrationWorker()
            self.calibration_worker.moveToThread(self.calibration_thread)
            self.calibration_thread.finished.connect(self.calibration_worker.deleteLater)
            self.calibration_requested.connect(self.calibration_worker.run)
            self.calibration_worker.progress.connect(self.on_calibration_progress)
            self.calibration_worker.done.connect(self.on_calibration_done)
            self.calibration_worker.failed.connect(self.on_calibration_failed)
            self.calibration_thread.start()
        self.calibration_requested.emit(frame,self.media_generation,source)

    # 仅更新当前校准世代的进度提示，忽略换源前的旧消息。
    def on_calibration_progress(self,generation,message):
        if self.calibration_busy and generation==self.media_generation:
            self.statusBar().showMessage("校准中 · "+message)

    # 核对世代后接受几何、恢复相关显示并按原流程保存；旧任务不得覆盖新来源。
    # 自动纠错报告 unchanged 时保留旧几何语义，不把提示与实际保存结果混淆。
    def on_calibration_done(self,generation,frame,result):
        if generation!=self.media_generation:return
        report=result.get('calibration_report')
        persistence_error=None
        if report:
            try:
                from src.calibration_state import save_automatic,write_json
                path=self.calibration_profile_path()
                profile=json.loads(path.read_text()) if path.exists() else None
                if report['decision'] in ('initial','updated') or report.get('scope_corrected'):
                    save_automatic(path.with_suffix('.auto.json'),frame,result,profile)
                history_path=path.with_suffix('.feedback.json')
                history=json.loads(history_path.read_text()) if history_path.exists() else {}
                history['reports']=(history.get('reports',[])+[report])[-6:]
                write_json(history_path,history)
            except (OSError,ValueError,KeyError,TypeError) as exc:persistence_error=str(exc)
        if not report:
            try:
                from src.calibration_state import save_automatic
                path=self.calibration_profile_path()
                profile=json.loads(path.read_text()) if path.exists() else None
                save_automatic(path.with_suffix('.auto.json'),frame,{**result,'calibration_report':{'decision':'manual_applied'}},profile)
            except (OSError,ValueError,KeyError,TypeError) as exc:persistence_error=str(exc)
        self._calibration_backup=None
        self.canvas.set_frame(frame)
        polygon=result['polygon']
        self.canvas.calibration_polygon=polygon
        self.canvas.recognition_polygon=result.get('scope_polygon') or result['pockets']['table'].get('scope_polygon') or polygon
        self.single_table_polygon=polygon
        if self.yolo_worker is not None:
            self.yolo_worker.calibration_polygon=polygon
            self.yolo_worker.scope_polygon=self.canvas.recognition_polygon
        self.canvas.calibrate_coordinates()
        self.last_pocket_result=result['pockets']
        self.canvas.set_pocket_result(result['pockets'])
        self.canvas.set_detections(result['detections'])
        self.start_camera_alignment()
        self.refresh_calibration_state()
        if persistence_error:self.calibration_state_label.setText('当前校准未保存，请查看底部错误信息')
        self.load_exam_marks()
        self.yolo_busy=False
        self.calibration_controls(False)
        self.detection_mode="idle"
        self.update_all_button()
        observed=sum(p['status']=='observed' for p in result['pockets']['pockets'])
        missing=sum(p['status']=='missing' for p in result['pockets']['pockets'])
        manual=sum(p['status']=='manual' for p in result['pockets']['pockets'])
        estimated=6-observed-missing-manual
        conclusion=(f"纠错完成：比较 {report['candidate_count']} 个候选，已修正边界" if report and report['decision']!='unchanged' else
                    (("未找到更可靠的修正，保留已保存的整套校准；再次校准将扩大搜索" if report['attempt']<4 else
                      "未找到更可靠的替代结果，已保留整套校准；重复点击不会提高精度") if report else
                    "人工校准已应用"))
        if report and report.get('scope_corrected'):
            conclusion='已恢复连续同色台面范围，黑色和木质外框排除，坐标内沿独立保留'
        comparison=''
        if report and report.get('before') and report['decision']=='updated':
            before,after=report['before'],report['after']
            comparison=f"（边缘黑色 {before['black_band']:.0%}→{after['black_band']:.0%}，袋口验证 {before['observed']}→{after['observed']}）"
        self.statusBar().showMessage(
            conclusion+comparison+f"：台面 1；袋口已验证 {observed}，人工确认 {manual}，估算 {estimated}，未定位 {missing}"
            + (" · 袋口尚不完整，请减少遮挡后重试" if observed+manual<6 else " · 点击播放继续")
            + (f" · 结果保存失败，下次打开无法保证恢复：{persistence_error}" if persistence_error else "")
            + (" · 学习样本已保存（待训练）" if result.get("learning_sample") else "")
            + (f" · 学习样本保存失败：{result['learning_sample_error']}" if result.get("learning_sample_error") else ""))
        if self.pot_button.isChecked():
            self.prepare_pot_detection()
            if getattr(self,'pot_resume_after_calibration',False):
                self.pot_resume_after_calibration=False
                self.play_video()
        self.canvas.update()

    # 保留原有效几何并恢复操作控件，明确显示失败，不以空结果覆盖好配置。
    def on_calibration_failed(self,generation,message):
        if generation!=self.media_generation:return
        backup=getattr(self,'_calibration_backup',None)
        if backup:
            self.canvas.set_frame(backup['frame'])
            for key,value in backup.items():
                if key not in ('frame','_main'):setattr(self.canvas,key,value)
            for key,value in backup.get('_main',{}).items():setattr(self,key,value)
            polygon=self.canvas.calibration_polygon
            self.single_table_polygon=polygon
            if self.yolo_worker is not None:
                self.yolo_worker.calibration_polygon=polygon
                self.yolo_worker.scope_polygon=self.canvas.recognition_polygon
            self.last_pocket_result=self.canvas.pocket_result
            self.canvas.pockets_available.emit(self.canvas.pocket_result is not None)
            self.canvas.calibrate_coordinates()
            self.start_camera_alignment()
            self.load_exam_marks()
            self.canvas.update()
        self._calibration_backup=None
        self.yolo_busy=False
        self.calibration_controls(False)
        self.statusBar().showMessage(f"校准未通过：{message}；原识别结果已保留，可再次校准或手动修正")

    # 切换坐标显示，必要时准备可靠几何；不因查看毫米坐标改变播放/暂停状态。
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

    # 递增世代并清理选择、轨迹、配准及进球会话；异步回包必须因此失效。
    def invalidate_results(self):
        self.camera_alignment=None;self.camera_geometry=None
        self.close_exam()
        self.canvas.exam_circles=[]
        self.calibration_state_label.setText("校准：尚未加载")
        self.reset_pot_session("来源或校准已改变")
        self.canvas.calibration_polygon=None
        self.canvas.recognition_polygon=None
        self.single_table_polygon=None
        if self.yolo_worker is not None:
            self.yolo_worker.calibration_polygon=None
            self.yolo_worker.scope_polygon=None
        self.track_recorder.close();self.canvas.track_paths={}
        self.selected_targets.clear();self.canvas.marked_ids.clear();self.pending_selection=None
        self.resume_after_selection=False;self.selection_busy=False
        self.detection_mode="idle"
        self.live_tracking.clear()
        self.live_selected_id=None
        self.canvas.coordinates=None
        self.canvas.coordinate_ball=None
        if self.canvas.coordinate_mode:self.coordinate_readout.setText("等待当前画面台面定位")
        self.ball_coordinate_readout.setText("球心：未选择")
        self.media_generation += 1
        self.last_pocket_result = None
        self.canvas.clear_pockets()
        self.yolo_generation += 1
        self.pending_all = None
        self.canvas.clear_selection()

    # 按当前来源列表更新上一项/下一项的边界状态和文件名。
    def update_navigation(self):
        images = self.media_mode == "images"
        position=self.image_index if images else self.video_index
        count=len(self.image_paths) if images else len(self.video_paths)
        self.previous_button.setEnabled(position>0)
        self.next_button.setEnabled(position+1<count)
        self.previous_shortcut.setEnabled(True)
        self.next_shortcut.setEnabled(True)
        self.previous_button.setText("← 上一张" if images else "← 上一个")
        self.next_button.setText("下一张 →" if images else "下一个 →")
        if images and self.image_index >= 0:
            self.image_info.setText(f"{self.image_index + 1} / {len(self.image_paths)} · {self.image_paths[self.image_index].name}")
        elif images:
            self.image_info.setText("未找到样本图片")
        else:
            self.image_info.setText(f"{self.video_index+1} / {len(self.video_paths)} · {self.video_path.name}")
        self.update_playback_button()
        self.update_all_button()

    # 先检查索引边界，再切换到目标视频或图片，复用统一失效流程。
    def navigate_media(self,step):
        if self.media_mode=="images":
            self.show_image(self.image_index+step);return
        index=self.video_index+step
        if not 0<=index<len(self.video_paths):return
        self.timer.stop()
        if self.capture is not None:self.capture.release()
        self.capture=None
        self.video_index=index;self.video_path=self.video_paths[index]
        self.switch_source(0)

    # 换源前释放旧视频和识别会话，再打开新来源；视频首帧显示后保持暂停。
    def switch_source(self, index):
        self.update_pot_count(clear=True);self.pot_reset_count_on_play=False
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
            self.restore_saved_calibration()
            self.playback_ready = True
            self.timer.setInterval(round(1000 / fps) if 1 <= fps <= 120 else 40)
            self.statusBar().showMessage(f"本地视频：{self.video_path.name} · 已就绪，点击播放开始")
        self.update_navigation()
        if getattr(self,'_calibration_restore_note',None):self.statusBar().showMessage(self._calibration_restore_note)

    # 读取当前样本图片并恢复其配置；读取失败也保持导航可用，不启动视频计时器。
    def show_image(self, index):
        if self.media_mode != "images" or not 0 <= index < len(self.image_paths):
            return
        self.timer.stop()
        self.invalidate_results()
        self.update_pot_count(clear=True)
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
            restored=self.restore_saved_calibration()
            self.playback_ready = True
            self.statusBar().showMessage(f"样本图片：{path.name} · " + (self._calibration_restore_note if restored else getattr(self,'_calibration_restore_note',None) or "点击“识别全部”或开启 YOLO 后点选"))
        self.update_navigation()

    # 按需创建/停止 worker 和 QThread；关闭识别与独立进球像素观察分开处理。
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
            self.resume_after_selection=False
            self.update_playback_button()
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
            self.yolo_worker.calibration_polygon=self.canvas.calibration_polygon
            self.yolo_worker.scope_polygon=self.canvas.recognition_polygon
            self.yolo_worker.moveToThread(self.yolo_thread)
            self.yolo_thread.finished.connect(self.yolo_worker.deleteLater)
            self.identify_requested.connect(self.yolo_worker.identify)
            self.identify_all_requested.connect(self.yolo_worker.identify_all)
            self.live_requested.connect(self.yolo_worker.identify_live)
            self.load_requested.connect(self.yolo_worker.load)
            self.yolo_worker.ready.connect(self.on_yolo_ready)
            self.yolo_worker.geometry_ready.connect(self.on_yolo_geometry)
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

    # 模型握手完成后恢复识别入口；是否继续提交由当前模式/播放状态决定。
    def on_yolo_ready(self):
        self.yolo_ready = True
        self.yolo_loading = False
        if self.calibration_busy:return
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

    # 集中判断当前是否满足视频连续推理条件，避免静态图片误进逐帧请求循环。
    def live_active(self):
        return self.media_mode=="video" and self.timer.isActive() and self.yolo_button.isChecked() and (self.detection_mode!="idle" or self.automatic_trails())

    # 同一时间最多保留一个模型请求；忙时不排队，空闲时取当前最新帧。
    # 记录请求 token、来源世代和帧号，响应返回后才能判断是否仍适用。
    def maybe_submit_live(self):
        if not self.live_active() or not self.yolo_ready or self.yolo_loading or self.yolo_busy or self.canvas.frame is None:
            return
        if not self.live_tracking.frames:
            self.live_tracking.push(self.live_frame_id,self.canvas.frame)
        self.live_token+=1
        self.live_inflight=(self.live_token,self.yolo_generation,self.live_frame_id,time.monotonic())
        payload=self.canvas.frame.copy()
        self.yolo_busy=True
        self.update_all_button()
        if self.detection_mode=="single" and not self.automatic_trails() and not self.pot_button.isChecked():
            targets=[]
            for key,target in self.selected_targets.items():
                target=dict(target);state=self.live_tracking.identities.states.get(key)
                if state is not None:
                    if state['appearance'] is not None:target['_appearance']=state['appearance'].tolist()
                    if state['status']=='lost':
                        dt=min(12,max(0,self.live_frame_id-state['frame']))
                        position=state['center']+state['velocity']*dt;half=state['size']/2
                        target['box']=[*(position-half).tolist(),*(position+half).tolist()]
                targets.append(target)
            if not targets:
                self.yolo_busy=False;self.live_inflight=None;self.update_all_button();return
            payload={"frame":payload,"target":targets[0]} if len(targets)==1 else {"frame":payload,"targets":targets}
        self.live_requested.emit(payload,self.live_token)

    # 只在现有几何允许时吸收模型附带的初始台面证据，避免覆盖已有人工/保存基准。
    def adopt_initial_geometry(self,detections):
        if self.canvas.calibration_polygon is not None:return
        table=next((d for d in detections if d.get('calibration_evidence')),None)
        if table is None:return
        evidence=table['calibration_evidence']
        metrics=evidence['metrics']
        if not metrics['reliable'] or metrics['observed']!=6:return
        path=self.calibration_profile_path()
        if path.exists():return  # Human annotations require explicit correction feedback.
        polygon=evidence['polygon']
        self.canvas.calibration_polygon=polygon;self.single_table_polygon=polygon
        self.canvas.recognition_polygon=evidence['pockets']['table'].get('scope_polygon') or polygon
        self.canvas.calibrate_coordinates()
        self.canvas.set_pocket_result(evidence['pockets']);self.last_pocket_result=evidence['pockets']
        if self.yolo_worker is not None:
            self.yolo_worker.calibration_polygon=polygon
            self.yolo_worker.scope_polygon=self.canvas.recognition_polygon
        self.load_exam_marks()
        report={'attempt':0,'candidate_count':1,'decision':'initial','strategy':'validated_first_detection',
                'before':None,'after':metrics,'reason':'首次自动识别通过台呢、黑边和六袋口验证'}
        try:
            from src.calibration_state import save_automatic
            save_automatic(path.with_suffix('.auto.json'),self.canvas.frame,
                           {'polygon':polygon,'pockets':evidence['pockets'],'calibration_report':report},None)
        except (OSError,ValueError) as exc:print(f'自动识别配置保存失败：{exc}',file=sys.stderr,flush=True)
        self.refresh_calibration_state()
        self.start_camera_alignment()

    # 核对请求世代后转交初始几何；独立于球体结果显示。
    def on_yolo_geometry(self,generation,table):
        if not self.calibration_busy and generation==self.yolo_generation and self.yolo_button.isChecked():
            self.adopt_initial_geometry([table])

    # 核对 token/来源并把延迟检测对齐到当前帧，再更新选择与轨迹。
    # 进球逐帧观察已经独立执行，不能将模型回包频率当作视频分析帧率。
    def on_live_result(self,token,detections):
        request=self.live_inflight
        if request is None or token!=request[0]:return
        self.live_inflight=None;self.yolo_busy=False
        self.update_all_button()
        if self.pending_selection is not None:
            frame,x,y,epoch=self.pending_selection;self.pending_selection=None
            if epoch==self.yolo_generation:
                resume=self.resume_after_selection
                self.identify_element(frame,x,y)
                self.resume_after_selection=resume
            return
        if not self.yolo_button.isChecked() or request[1]!=self.yolo_generation:
            self.maybe_submit_live();return
        # Never apply a result older than the bounded frame history.
        if self.live_tracking.replace(request[2],detections):
            self.adopt_initial_geometry(detections)
            if self.detection_mode=="single" and not self.automatic_trails():
                self.live_tracking.detections=[item for item in self.live_tracking.detections if item.get("track_id") in self.selected_targets]
            self.canvas.set_detections(self.live_tracking.detections)
            self.render_live_selection()
            self.sync_tracks()
            now=time.monotonic()
            rate=1/(now-self.live_last_result_time) if self.live_last_result_time else 0.
            self.live_last_result_time=now
            suffix=f" · 推理更新 {rate:.1f} 次/秒" if rate else ""
            mode="播放与推理独立"
            self.statusBar().showMessage(f"YOLO 实时识别：{len(detections)} 个目标{suffix} · {mode}")
        # The next video tick submits only the newest frame; no FIFO of images.

    # 释放该请求的忙碌状态并提示错误，保留播放主循环可继续工作。
    def on_live_error(self,token,message):
        if self.calibration_busy:return
        request=self.live_inflight
        if request is None or token!=request[0]:return
        self.live_inflight=None;self.yolo_busy=False
        self.live_tracking.detections=[];self.canvas.clear_selection()
        if self.yolo_worker and self.yolo_worker.process and self.yolo_worker.process.poll() is not None:
            self.yolo_ready=False
        self.update_all_button()
        if request[1]==self.yolo_generation:self.statusBar().showMessage(f"YOLO 实时识别失败：{message}")

    # 对可命中的已有目标切换选择集合，保留身份 ID，而非按每帧框顺序认球。
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

    # 将当前选择映射到最新跟踪检测；跟丢时不把旧框当作持续实测结果。
    def render_live_selection(self):
        self.canvas.show_ball_boxes=self.detection_mode=="all" or self.detection_mode=="single" and not self.automatic_trails()
        if self.automatic_trails():
            self.canvas.selected_box=None;self.canvas.selected_polygon=None;self.canvas.marked_ids.clear()
            return
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
            self.canvas.calibrate_coordinates()
            position=self.canvas.coordinates.ball_center(item['box']) if self.canvas.coordinates else None
            self.canvas.coordinate_ball=position
            self.ball_coordinate_readout.setText(f"已选 {len(self.selected_targets)} 个 | {label}：X = {position[0]:.1f} mm，Y = {position[1]:.1f} mm" if position else f"已选 {len(self.selected_targets)} 个目标")
        if item["name"]=="billiards_table":self.ball_coordinate_readout.setText(f"已选 {len(self.selected_targets)} 个（当前：台球桌）")
        self.canvas.update()

    # 按任务忙碌状态更新袋口入口，避免并行重复检测。
    def update_pocket_button(self):
        self.pocket_button.setText("袋口检测中…" if self.pocket_busy else "自动检测袋口")
        self.pocket_button.setEnabled(self.playback_ready and not self.pocket_busy)

    # 复用当前显式校准入口处理袋口，不为每个视频帧重新定位整桌。
    def detect_pockets(self):
        if self.calibration_busy or not self.playback_ready or self.canvas.frame is None:return
        self.calibrate_table()

    # 只接受当前来源世代的袋口结果，更新画布及依赖该几何的功能状态。
    def on_pocket_result(self, generation, result):
        if self.calibration_busy:return
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

    # 当前袋口任务失败时恢复控件并显示原因，不发布一个假成功的六袋结果。
    def on_pocket_error(self, generation, message):
        if self.calibration_busy:return
        self.pocket_busy = False
        self.update_pocket_button()
        if generation == self.media_generation:
            self.statusBar().showMessage(f"袋口检测或保存失败：{message}")

    # 根据来源、模型就绪和当前选择模式同步“全部识别”入口文字/可用性。
    def update_all_button(self):
        if self.calibration_busy:return
        self.update_pocket_button()
        self.canvas.show_ball_boxes=self.detection_mode=="all" or self.detection_mode=="single" and not self.automatic_trails()
        active=self.detection_mode=="all"
        self.identify_all_button.setText("取消识别" if active else "全部识别")
        self.identify_all_button.setEnabled(self.playback_ready and (active or not self.yolo_loading and not self.yolo_busy))

    # 显式切换到整桌识别；播放中走连续模式，静态画面只提交一次请求。
    def identify_all_elements(self):
        if self.calibration_busy:return
        if self.detection_mode=="all":
            self.yolo_generation+=1
            self.pending_all=None;self.pending_selection=None
            self.detection_mode="idle"
            self.trails_button.setCurrentIndex(4)
            self.selected_targets.clear();self.live_selected_id=None
            self.live_tracking.clear()
            self.canvas.clear_selection();self.canvas.track_paths={};self.canvas.marked_ids.clear()
            self.canvas.coordinate_ball=None
            self.ball_coordinate_readout.setText("球心：未选择")
            self.canvas.update();self.update_all_button()
            self.statusBar().showMessage("已取消全部识别")
            return
        if not self.playback_ready or self.canvas.frame is None or self.yolo_busy:return
        self.yolo_generation+=1
        self.detection_mode="all"
        self.live_tracking.clear()
        self.selected_targets.clear();self.canvas.marked_ids.clear()
        self.live_selected_id=None
        self.canvas.clear_selection();self.canvas.track_paths={}
        self.ball_coordinate_readout.setText("球心：未选择")
        if not self.yolo_button.isChecked():self.yolo_button.setChecked(True)
        if self.media_mode=="video" and self.timer.isActive():
            self.update_all_button();self.maybe_submit_live()
            return
        frame=self.canvas.frame.copy()
        if not self.yolo_ready and not self.yolo_loading:self.toggle_yolo(True)
        if self.yolo_ready:self.submit_all(frame,self.yolo_generation)
        else:self.pending_all=(frame,self.yolo_generation)
        self.update_all_button()

    # 设置忙碌状态并复制当前帧发给后台，保证推理对应提交时的画面。
    def submit_all(self, frame, generation):
        self.yolo_busy = True
        self.update_all_button()
        self.statusBar().showMessage("正在识别当前画面中的全部目标…")
        self.identify_all_requested.emit(frame, generation)

    # 核对世代后显示全目标结果；过时响应不更新画布或新的选择。
    def on_yolo_all_result(self, generation, detections):
        if self.calibration_busy:return
        self.yolo_busy = False
        self.update_all_button()
        if not self.yolo_button.isChecked() or generation != self.yolo_generation:
            return
        self.adopt_initial_geometry(detections)
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

    # 将有效原图点击坐标与冻结帧交给后台；播放中的选择优先复用已有目标。
    def identify_element(self, frame, x, y):
        if self.calibration_busy:return
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
        self.yolo_busy=True;self.selection_busy=True;self.update_all_button()
        self.ball_coordinate_readout.setText(f"已选 {len(self.selected_targets)} 个 · 正在识别新目标…")
        self.identify_requested.emit(frame,x,y,self.yolo_generation)

    # 核对世代并展示点选名称、框和球心；未命中时显示未识别而非最近球。
    def on_yolo_result(self, generation, name, confidence, box=None, polygon=None):
        if self.calibration_busy:return
        self.selection_busy=False
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
            if self.resume_after_selection:
                self.resume_after_selection=False;self.play_video()
        else:
            self.resume_after_selection=False
            self.update_playback_button()
            self.ball_coordinate_readout.setText(f"已选 {len(self.selected_targets)} 个 · 点击处未识别到目标")

    # 清理当前静态请求状态并显示异常，避免错误后按钮永久处于忙碌。
    def on_yolo_error(self, generation, message):
        if self.calibration_busy:return
        self.selection_busy=False;self.resume_after_selection=False
        self.update_playback_button()
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

    # 由实际计时器/视频可用状态决定单按钮文字和图标，而非另存易失同步标记。
    def update_playback_button(self):
        playing=self.timer.isActive() or self.resume_after_selection
        self.play_button.setText("暂停" if playing else "播放")
        self.play_button.setAccessibleName("暂停视频" if playing else "播放视频")
        self.play_button.setEnabled(self.media_mode=="video" and self.playback_ready and not self.calibration_busy)

    # 同一入口根据计时器状态调用暂停或播放，保持单按钮行为一致。
    def toggle_playback(self):
        if self.timer.isActive() or self.resume_after_selection:self.pause_video()
        else:self.play_video()

    # 恢复视频时按原规则更新选择和会话；只启动媒体计时器，不同步等待模型推理。
    def play_video(self):
        if self.calibration_busy:return
        if self.pot_reset_count_on_play:
            self.update_pot_count(clear=True);self.pot_reset_count_on_play=False
        if self.pot_button.isChecked():self.prepare_pot_detection()
        if self.selection_busy or self.pending_selection is not None:
            self.resume_after_selection=True
            self.update_playback_button()
            self.statusBar().showMessage("正在确认所选球，完成后自动播放")
            return
        self.ball_coordinate_readout.setText("球心：未选择")
        if self.media_mode == "images" or not self.playback_ready:
            return
        self.yolo_generation += 1
        self.media_generation += 1
        if not self.canvas.calibration_polygon and not self.pot_button.isChecked():
            self.last_pocket_result = None
            self.canvas.clear_pockets()
        self.pending_all = None
        self.update_all_button()
        self.canvas.clear_selection()
        if self.detection_mode!="single":
            self.live_tracking.clear();self.live_selected_id=None
        self.timer.start()
        self.maybe_submit_live()
        self.update_playback_button()
        self.statusBar().showMessage(f"本地视频：{self.video_path.name} · 正在播放")

    # 停止解码计时器并刷新按钮；暂停不推进视频时间，也不应增加入袋消失证据。
    def pause_video(self):
        self.resume_after_selection=False
        if self.media_mode == "images" or not self.playback_ready:
            return
        self.timer.stop()
        self.update_playback_button()
        self.statusBar().showMessage(f"本地视频：{self.video_path.name} · 已暂停")

    # 视频打开/读取异常的统一界面反馈，禁用依赖不可用视频的操作。
    def show_error(self, message):
        self.timer.stop()
        self.playback_ready = False
        self.pending_all = None
        self.update_all_button()
        self.update_playback_button()
        self.canvas.message = message
        self.canvas.update()
        self.statusBar().showMessage("图片加载失败" if self.media_mode == "images" else "视频播放失败")
        print(message, file=sys.stderr, flush=True)

    # 主线程逐帧顺序：解码、显示/轨迹更新、相机对齐、进球像素观察，再尝试异步推理。
    # EOF 回到首帧并暂停，清理过期任务；不能把循环首帧拼接到上一段入袋轨迹。
    def next_frame(self):
        if self.calibration_busy:return
        self.ball_coordinate_readout.setText("球心：未选择")
        self.media_generation += 1
        if not self.canvas.calibration_polygon and not self.pot_button.isChecked():self.last_pocket_result = None
        ok, frame = self.capture.read()
        ended=not ok
        ended_pot_text=None
        if ended:
            self.timer.stop()
            if self.pot_adapter is not None:
                engine=self.pot_adapter.engine
                if not engine.completed:
                    results=engine.finish(force_unknown='recording_ended_before_final_review')
                    self.update_pot_count(results)
                    if self.pot_recording:self.pot_recording.write(engine.snapshot(),results,self.canvas.frame)
                words={'POTTED':'已进袋','NOT_POTTED':'未进袋'}
                ended_pot_text='本段结束 · '+' | '.join(title+'：'+words.get(engine.states[ball].state.value,'无法确认，需要人工复核')
                    for ball,title in (('black_ball','黑球'),('white_ball','白球')))+' · 重播将重新观察'
            self.invalidate_results()
            self.capture.set(cv2.CAP_PROP_POS_FRAMES,0)
            ok,frame=self.capture.read()
        if not ok:
            self.canvas.image = QImage()
            self.show_error("无法读取视频帧，请检查视频文件是否完整。")
            return
        if ended:
            self.canvas.set_frame(frame)
            self.restore_saved_calibration()
            self.update_navigation()
            self.statusBar().showMessage("播放完毕 · 已回到首帧，点击播放可重播")
            if ended_pot_text:self.pot_readout.setText(ended_pot_text)
            self.pot_reset_count_on_play=True
            return
        # 顺序很重要：先更新当前工作几何，再让本帧实测球进入事件适配器。
        self.update_camera_alignment(frame)
        self.observe_current_pot_frame(frame)
        self.live_frame_id+=1
        visual_tracking=self.yolo_button.isChecked() and (not self.pot_button.isChecked() or self.automatic_trails() or self.detection_mode!="idle")
        tracked=self.live_tracking.push(self.live_frame_id,frame) if visual_tracking else []
        self.canvas.set_frame(frame,keep_pockets=bool(self.canvas.calibration_polygon) or self.pot_button.isChecked())
        if self.yolo_button.isChecked():
            self.canvas.set_detections(tracked)
            self.render_live_selection()
            self.sync_tracks()
            self.maybe_submit_live()

    # 先停止计时与记录，再关闭模型进程及各工作线程，最后交给 Qt 释放窗口。
    # 现有 QThread.wait/记录器关闭仍可能等待任务完成，不在本轮注释修改中增加退出超时。
    def closeEvent(self, event):
        self.close_exam()
        self.reset_pot_session("软件关闭")
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
        if self.calibration_thread is not None:
            self.calibration_thread.quit()
            self.calibration_thread.wait()
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

"""Explicit point confirmation when a camera's automatic geometry is unsuitable."""
# 人工四角/六袋编辑器；画布支持缩放和平移，结果以原图归一化坐标保存。
# 屏幕坐标只用于交互，不能直接写成校准点；接受前必须通过 manual_geometry 校验。

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPolygonF, QKeySequence
from PyQt5.QtWidgets import QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QShortcut

LABELS=['台呢内沿左上角','台呢内沿右上角','台呢内沿右下角','台呢内沿左下角',
        '第 1 个袋口中心','第 2 个袋口中心','第 3 个袋口中心','第 4 个袋口中心','第 5 个袋口中心','第 6 个袋口中心']


class PointCanvas(QWidget):
    changed=pyqtSignal()
    zoom_changed=pyqtSignal(float)

    # 建立原帧显示副本、编辑点与缩放/拖动状态；图片绘制不修改原始输入。
    def __init__(self, frame):
        super().__init__()
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w=rgb.shape[:2]
        self.image=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()
        self.points=[]
        self.zoom=1.
        self.pan=QPointF()
        self.pan_mode=False
        self.press_position=None
        self.last_position=None
        self.dragged=False
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(300,300)
        self.setCursor(Qt.CrossCursor)
        self.setAccessibleName('校准图像，按提示点击内沿四角和六个袋口；可撤销或取消')

    # 返回当前缩放和平移后的图像矩形，所有点选逆变换都基于它。
    def area(self):
        size=self.image.size().scaled(self.size(),Qt.KeepAspectRatio)
        w,h=size.width()*self.zoom,size.height()*self.zoom
        return QRectF((self.width()-w)/2+self.pan.x(),(self.height()-h)/2+self.pan.y(),w,h)

    # 限制平移，使缩放后图像仍可在可用区域内操作。
    def clamp_pan(self):
        size=self.image.size().scaled(self.size(),Qt.KeepAspectRatio)
        x=abs(size.width()*self.zoom-self.width())/2
        y=abs(size.height()*self.zoom-self.height())/2
        self.pan=QPointF(max(-x,min(x,self.pan.x())),max(-y,min(y,self.pan.y())))

    # 更新缩放并保持指定屏幕锚点对应的图像位置，避免滚轮缩放时校准点跳动。
    def set_zoom(self,value,anchor=None):
        anchor=anchor or QPointF(self.width()/2,self.height()/2)
        old=self.area()
        nx,ny=(anchor.x()-old.left())/old.width(),(anchor.y()-old.top())/old.height()
        self.zoom=max(1.,min(12.,value))
        self.pan=QPointF()
        new=self.area()
        self.pan=QPointF(anchor.x()-new.left()-nx*new.width(),anchor.y()-new.top()-ny*new.height())
        self.clamp_pan();self.zoom_changed.emit(self.zoom);self.update()

    # 恢复适配显示与平移状态，不删除已经选取的校准点。
    def fit_image(self):
        self.zoom=1.;self.pan=QPointF()
        self.zoom_changed.emit(self.zoom);self.update()

    # 切换平移交互，避免拖动画面被误认为新角点。
    def set_pan_mode(self,enabled):
        self.pan_mode=enabled
        self.setCursor(Qt.OpenHandCursor if enabled else Qt.CrossCursor)

    # 以鼠标位置为缩放锚点处理滚轮，触发重绘而非更改实际校准坐标。
    def wheelEvent(self,event):
        delta=event.angleDelta().y() or event.pixelDelta().y()
        if delta:self.set_zoom(self.zoom*pow(1.2,delta/120),event.posF())
        event.accept()

    # 区分平移与取点；只有有效图像范围中的点会转换成编辑坐标。
    def mousePressEvent(self,event):
        if event.button() in (Qt.LeftButton,Qt.MiddleButton):
            self.setFocus()
            self.press_position=event.localPos();self.last_position=event.localPos();self.dragged=False
            if self.pan_mode or event.button()==Qt.MiddleButton:self.setCursor(Qt.ClosedHandCursor)

    # 拖动平移时只改变显示偏移，校准点仍保留在图像空间。
    def mouseMoveEvent(self,event):
        if self.press_position is not None:
            if (event.localPos()-self.press_position).manhattanLength()>5:self.dragged=True
            if self.dragged or self.pan_mode:
                self.pan+=event.localPos()-self.last_position
                self.clamp_pan();self.setCursor(Qt.ClosedHandCursor);self.update()
            self.last_position=event.localPos()

    # 只有未发生拖动的左键释放才新增归一化点；随后结束平移状态，避免拖动被误记为取点。
    def mouseReleaseEvent(self,event):
        area=self.area()
        if (event.button()==Qt.LeftButton and self.press_position is not None and not self.dragged
                and not self.pan_mode and area.contains(event.localPos()) and len(self.points)<10):
            p=event.localPos()
            self.points.append([(p.x()-area.left())/area.width(),(p.y()-area.top())/area.height()])
            self.changed.emit()
        self.press_position=None;self.last_position=None
        self.setCursor(Qt.OpenHandCursor if self.pan_mode else Qt.CrossCursor);self.update()

    # 控件尺寸变化后调整可见区域，保持图像和点使用同一变换。
    def resizeEvent(self,event):
        self.clamp_pan();super().resizeEvent(event)

    # 方向键只平移视图，其它按键交给 Qt；不会直接改变已选校准点。
    def keyPressEvent(self,event):
        directions={Qt.Key_Left:QPointF(40,0),Qt.Key_Right:QPointF(-40,0),
                    Qt.Key_Up:QPointF(0,40),Qt.Key_Down:QPointF(0,-40)}
        if event.key() in directions:
            self.pan+=directions[event.key()];self.clamp_pan();self.update();event.accept()
        else:super().keyPressEvent(event)

    # 绘制缩放图像、点位及连线；不在绘制回调中保存配置。
    def paintEvent(self,event):
        painter=QPainter(self);painter.fillRect(self.rect(),Qt.black)
        area=self.area();painter.drawImage(area,self.image)
        mapped=[QPointF(area.left()+x*area.width(),area.top()+y*area.height()) for x,y in self.points]
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor('#00E5FF'),2));painter.setBrush(Qt.NoBrush)
        if len(mapped)>=4:painter.drawPolygon(QPolygonF(mapped[:4]))
        for i,point in enumerate(mapped):
            painter.drawEllipse(point,5,5)
            text=f'{i+1} {LABELS[i]}'
            width=painter.fontMetrics().horizontalAdvance(text)+10
            if not self.rect().contains(point.toPoint()):continue
            x=min(max(3,point.x()+8),max(3,self.width()-width-3))
            y=min(max(20,point.y()-9),self.height()-5)
            painter.fillRect(QRectF(x-3,y-17,width,22),QColor('#101820'))
            painter.drawText(QPointF(x,y),text)


class CalibrationEditor(QDialog):
    # 设置角点/六袋编辑步骤，自动结果仅作可修改起点，最终以用户确认为准。
    def __init__(self,frame,parent=None):
        super().__init__(parent)
        self.setWindowTitle('手动校准：确认台面与六个袋口')
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(850,850)
        self.frame=frame
        self.clear_saved=False
        layout=QVBoxLayout(self)
        self.hint=QLabel();self.hint.setWordWrap(True);self.hint.setMinimumHeight(55)
        layout.addWidget(self.hint)
        self.canvas=PointCanvas(frame)
        tools=QHBoxLayout();layout.addLayout(tools)
        zoom_out=QPushButton('− 缩小');zoom_in=QPushButton('+ 放大');fit=QPushButton('适应窗口')
        move=QPushButton('移动画面');move.setCheckable(True)
        maximize=QPushButton('最大化 / 还原')
        self.zoom_label=QLabel('100%')
        for button in (zoom_out,zoom_in,fit,move,maximize):button.setMinimumHeight(44);tools.addWidget(button)
        tools.addWidget(self.zoom_label)
        zoom_in.clicked.connect(lambda:self.canvas.set_zoom(self.canvas.zoom*1.5))
        zoom_out.clicked.connect(lambda:self.canvas.set_zoom(self.canvas.zoom/1.5))
        fit.clicked.connect(self.canvas.fit_image);move.toggled.connect(self.canvas.set_pan_mode)
        maximize.clicked.connect(lambda:self.showNormal() if self.isMaximized() else self.showMaximized())
        self.canvas.zoom_changed.connect(lambda value:self.zoom_label.setText(f'{value*100:.0f}%'))
        self.zoom_shortcuts=[]
        for key,action in [('=',lambda:self.canvas.set_zoom(self.canvas.zoom*1.5)),
                           ('+',lambda:self.canvas.set_zoom(self.canvas.zoom*1.5)),
                           ('-',lambda:self.canvas.set_zoom(self.canvas.zoom/1.5)),('0',self.canvas.fit_image)]:
            shortcut=QShortcut(QKeySequence(key),self);shortcut.activated.connect(action);self.zoom_shortcuts.append(shortcut)
        tip=QLabel('滚轮缩放 · 拖动画面查看局部 · 单击标注 · 方向键移动 · 0 恢复全图')
        tip.setWordWrap(True);layout.addWidget(tip)
        layout.addWidget(self.canvas,1)
        row=QHBoxLayout();layout.addLayout(row)
        self.undo=QPushButton('撤销上一步');reset=QPushButton('重新点选')
        cancel=QPushButton('取消');self.save=QPushButton('保存并重新识别')
        automatic=QPushButton('恢复自动校准')
        automatic.clicked.connect(self.restore_automatic)
        for button in (self.undo,reset,automatic,cancel,self.save):button.setMinimumHeight(44);row.addWidget(button)
        self.undo.clicked.connect(self.undo_point);reset.clicked.connect(self.reset_points)
        cancel.clicked.connect(self.reject);self.save.clicked.connect(self.validate_points)
        self.canvas.changed.connect(self.refresh);self.refresh()

    # 根据已取点数量刷新步骤、提示和接受按钮。
    def refresh(self):
        n=len(self.canvas.points)
        self.save.setEnabled(n==10);self.undo.setEnabled(n>0)
        self.hint.setText((f'第 {n+1}/10 步：点击{LABELS[n]}。' + ('四角选台呢与库边的交界，不选木框。' if n<4 else '六个袋口顺序不限，每个袋口只点一次。')) if n<10 else
                          '已点选 4 个内沿角点和 6 个袋口。检查蓝色边界后保存；之后重新检测球体，点选结果按当前来源保存。')

    # 撤销最后一个编辑点，允许逐步纠正取点顺序。
    def undo_point(self):
        if self.canvas.points:self.canvas.points.pop()
        self.refresh();self.canvas.update()

    # 清空本次编辑点，不修改已保存的旧校准。
    def reset_points(self):
        self.canvas.points=[];self.refresh();self.canvas.update()

    # 用几何模块验证四角/六袋，再允许接受；有十个点不代表它们一定有效。
    def validate_points(self):
        try:
            from src.manual_calibration import manual_geometry
            manual_geometry(self.frame,self.profile())
        except ValueError as exc:
            self.hint.setText(str(exc)+'；请撤销修改或重新点选。')
            return
        self.accept()

    # 将自动建议作为编辑起点重新载入，仍需用户确认。
    def restore_automatic(self):
        self.clear_saved=True
        self.accept()

    # 把当前有效点位转成可保存的归一化人工配置。
    def profile(self):
        return {'version':1,'corners':self.canvas.points[:4],'pockets':self.canvas.points[4:]}

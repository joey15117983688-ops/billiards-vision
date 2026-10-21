"""Snapshot circle editor; marks live in normalized calibrated table coordinates."""
from copy import deepcopy
import math
import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, QLabel
from src.table_coordinates import TableCoordinates


def validate_marks(data):
    if not isinstance(data,list) or len(data)>200:raise ValueError('无效的考题标注')
    result=[]
    for mark in data:
        values=[float(mark[k]) for k in ('u','v','diameter')]
        if not all(math.isfinite(x) for x in values) or not all(0<=x<=1 for x in values[:2]) or not 1<=values[2]<=80:
            raise ValueError('考题标注超出范围')
        result.append(dict(zip(('u','v','diameter'),values)))
    return result


def circle_geometry(coordinates,mark):
    """Use an image-plane circle, matching the actual snapshot without stretching."""
    d=coordinates.dimensions
    center=coordinates.pixels([[mark['u']*d.x_mm,mark['v']*d.y_mm]])[0]
    corners=coordinates.corners
    short=min(np.linalg.norm(corners[(i+1)%4]-corners[i]) for i in range(4))
    radius=short*mark['diameter']/200
    points=[[center[0]+radius*math.cos(t),center[1]+radius*math.sin(t)] for t in np.linspace(0,2*math.pi,81)]
    return center,points


def draw_marks(painter,coordinates,marks,map_point,selected=-1):
    painter.save();painter.setRenderHint(QPainter.Antialiasing);painter.setBrush(Qt.NoBrush)
    font=painter.font();font.setPixelSize(16);font.setBold(True);painter.setFont(font)
    for index,mark in enumerate(marks):
        center,points=circle_geometry(coordinates,mark)
        painter.setPen(QPen(QColor('#FFEB3B' if index!=selected else '#00E5FF'),3))
        painter.drawPolyline(QPolygonF([map_point(x,y) for x,y in points]))
        point=map_point(*center)
        # Keep the number upright and centred in both video orientations.
        label=str(index+1)
        metrics=painter.fontMetrics()
        baseline=QPointF(point.x()-metrics.horizontalAdvance(label)/2,
                         point.y()+(metrics.ascent()-metrics.descent())/2)
        text=QPainterPath();text.addText(baseline,font,label)
        painter.setPen(QPen(QColor('#111111'),3,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin))
        painter.setBrush(QColor('#FFEB3B' if index!=selected else '#00E5FF'))
        painter.drawPath(text)
        painter.fillPath(text,QColor('#FFEB3B' if index!=selected else '#00E5FF'))
        painter.setBrush(Qt.NoBrush)
    painter.restore()


class ExamCanvas(QWidget):
    changed=pyqtSignal(object)
    selected=pyqtSignal(int)

    def __init__(self,frame,polygon,dimensions,marks,parent=None):
        super().__init__(parent)
        self.coordinates=TableCoordinates(polygon,dimensions)
        self.marks=deepcopy(marks);self.selection=-1;self.diameter=10.;self.anchor=None;self.drag=None
        self.setMinimumSize(320,260);self.setCursor(Qt.CrossCursor);self.setMouseTracking(True)
        self.setAccessibleName('考题球桌画圆区域')
        x,y,w,h=cv2.boundingRect(np.float32(polygon))
        x=max(0,x);y=max(0,y);w=min(w,frame.shape[1]-x);h=min(h,frame.shape[0]-y)
        self.origin=(x,y)
        rgb=cv2.cvtColor(np.ascontiguousarray(frame[y:y+h,x:x+w]),cv2.COLOR_BGR2RGB)
        self.image=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()

    def area(self):
        size=self.image.size();size.scale(self.size(),Qt.KeepAspectRatio)
        return QRectF((self.width()-size.width())/2,(self.height()-size.height())/2,size.width(),size.height())

    def mapped(self,x,y):
        r=self.area()
        return QPointF(r.x()+(x-self.origin[0])*r.width()/self.image.width(),r.y()+(y-self.origin[1])*r.height()/self.image.height())

    def raw(self,point):
        r=self.area()
        return (self.origin[0]+(point.x()-r.x())*self.image.width()/r.width(),self.origin[1]+(point.y()-r.y())*self.image.height()/r.height())

    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect(),Qt.black);p.drawImage(self.area(),self.image)
        draw_marks(p,self.coordinates,self.marks,self.mapped,self.selection)

    def publish(self):
        self.update();self.changed.emit(deepcopy(self.marks))

    def hit_circle(self,point):
        # Hit tolerances are in screen pixels, independent of crop/zoom scale.
        for i in reversed(range(len(self.marks))):
            center,points=circle_geometry(self.coordinates,self.marks[i])
            screen=self.mapped(*center)
            radius=abs(self.mapped(*points[0]).x()-screen.x())
            distance=math.hypot(point.x()-screen.x(),point.y()-screen.y())
            tolerance=min(7.,max(3.,radius*.25))
            if distance>radius*.5 and abs(distance-radius)<=tolerance:return i,'resize'
            if distance<radius:return i,'move'
        return None

    def mousePressEvent(self,event):
        if event.button()!=Qt.LeftButton:return
        x,y=self.raw(event.localPos())
        hit=self.hit_circle(event.localPos())
        if hit is not None:
            self.selection,action=hit
            mark=self.marks[self.selection]
            center,points=circle_geometry(self.coordinates,mark)
            self.anchor=(x,y)
            self.drag={'action':action,'center':center,'mark':deepcopy(mark),
                       'distance':math.hypot(x-center[0],y-center[1])}
            self.setCursor(Qt.ClosedHandCursor if action=='move' else Qt.SizeFDiagCursor)
            self.selected.emit(self.selection);self.update();return
        if not self.area().contains(event.localPos()):return
        mm=self.coordinates.millimetres(x,y)
        if mm is None or len(self.marks)>=200:return
        d=self.coordinates.dimensions
        self.marks.append({'u':mm[0]/d.x_mm,'v':mm[1]/d.y_mm,'diameter':self.diameter})
        self.selection=len(self.marks)-1;self.anchor=(x,y)
        self.drag={'action':'create','center':[x,y],'mark':deepcopy(self.marks[-1]),'distance':0.}
        self.selected.emit(self.selection);self.publish()

    def mouseMoveEvent(self,event):
        if self.anchor is None or self.drag is None or not event.buttons() & Qt.LeftButton:
            hit=self.hit_circle(event.localPos())
            self.setCursor(Qt.CrossCursor if hit is None else Qt.OpenHandCursor if hit[1]=='move' else Qt.SizeFDiagCursor)
            return
        x,y=self.raw(event.localPos())
        pixels=self.area().width()/self.image.width()
        if math.hypot(x-self.anchor[0],y-self.anchor[1])*pixels<3:return
        candidate=deepcopy(self.drag['mark'])
        center=self.drag['center']
        corners=self.coordinates.corners
        if self.drag['action']=='move':
            position=np.array([center[0]+x-self.anchor[0],center[1]+y-self.anchor[1]])
            if cv2.pointPolygonTest(corners,tuple(position),False)<0:
                # Clamp the centre to the nearest table edge instead of losing it.
                closest=[]
                for a,b in zip(corners,np.roll(corners,-1,axis=0)):
                    edge=b-a;t=np.clip(np.dot(position-a,edge)/np.dot(edge,edge),0,1)
                    closest.append(a+t*edge)
                position=min(closest,key=lambda p:np.linalg.norm(p-position))
            mm=cv2.perspectiveTransform(np.float32([[position]]),self.coordinates.image_to_mm)[0,0]
            d=self.coordinates.dimensions
            candidate.update(u=float(np.clip(mm[0]/d.x_mm,0,1)),v=float(np.clip(mm[1]/d.y_mm,0,1)))
        else:
            short=min(np.linalg.norm(corners[(i+1)%4]-corners[i]) for i in range(4))
            distance=math.hypot(x-center[0],y-center[1])
            radius=distance if self.drag['action']=='create' else short*candidate['diameter']/200+distance-self.drag['distance']
            candidate['diameter']=float(np.clip(200*radius/short,1,80))
        if candidate!=self.marks[self.selection]:
            self.marks[self.selection]=candidate
            self.selected.emit(self.selection);self.publish()

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.LeftButton:
            self.anchor=None;self.drag=None
            hit=self.hit_circle(event.localPos())
            self.setCursor(Qt.CrossCursor if hit is None else Qt.OpenHandCursor if hit[1]=='move' else Qt.SizeFDiagCursor)


class ExamDialog(QDialog):
    changed=pyqtSignal(object)

    def __init__(self,frame,polygon,dimensions,marks,parent=None):
        super().__init__(parent)
        self.setWindowTitle('考题');self.resize(900,650)
        layout=QVBoxLayout(self);tools=QHBoxLayout();layout.addLayout(tools)
        tool=QPushButton('画圆');tool.setMinimumSize(80,44);tool.setToolTip('开始放置新圆圈');tool.clicked.connect(self.new_circle);tools.addWidget(tool)
        tools.addWidget(QLabel('直径（台面短边 %）：'))
        self.size_input=QDoubleSpinBox();self.size_input.setRange(1,80);self.size_input.setValue(10);self.size_input.setSuffix(' %');self.size_input.setMinimumHeight(44)
        self.size_input.setAccessibleName('圆圈直径，占台面短边百分比');tools.addWidget(self.size_input)
        for text,callback in [('删除选中',self.delete_selected),('清空',self.clear)]:
            button=QPushButton(text);button.setMinimumSize(90,44);button.clicked.connect(callback);tools.addWidget(button)
        tools.addStretch()
        self.canvas=ExamCanvas(frame,polygon,dimensions,marks,self);layout.addWidget(self.canvas,1)
        layout.addWidget(QLabel('空白处点击画圆；拖动圆内移动位置，拖动圆边调整大小，也可输入直径。视频中实时同步。'))
        self.canvas.changed.connect(self.changed);self.canvas.selected.connect(self.select)
        self.size_input.valueChanged.connect(self.resize_circle)

    def new_circle(self):
        self.canvas.selection=-1;self.canvas.anchor=None;self.canvas.drag=None;self.canvas.update()

    def select(self,index):
        self.size_input.blockSignals(True);self.size_input.setValue(self.canvas.marks[index]['diameter']);self.size_input.blockSignals(False)

    def resize_circle(self,value):
        self.canvas.diameter=value
        if self.canvas.selection>=0:
            self.canvas.marks[self.canvas.selection]['diameter']=value;self.canvas.publish()

    def delete_selected(self):
        if self.canvas.selection>=0:
            self.canvas.marks.pop(self.canvas.selection);self.canvas.selection=-1;self.canvas.anchor=None;self.canvas.drag=None;self.canvas.publish()

    def clear(self):
        self.canvas.marks=[];self.canvas.selection=-1;self.canvas.anchor=None;self.canvas.drag=None;self.canvas.publish()

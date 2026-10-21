import numpy as np
import pytest
from PyQt5.QtCore import Qt,QPoint,QPointF
from PyQt5.QtGui import QMouseEvent,QWheelEvent
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from src.calibration_editor import CalibrationEditor


def editor():
    app=QApplication.instance() or QApplication([])
    window=CalibrationEditor(np.zeros((600,1000,3),np.uint8))
    window.show();app.processEvents()
    return app,window


def test_zoomed_panned_click_is_saved_in_original_image_coordinates():
    app,window=editor();canvas=window.canvas
    canvas.set_zoom(4.)
    canvas.pan=QPointF(-150,110);canvas.clamp_pan()
    rect=canvas.area()
    expected=[.6,.45]
    point=QPoint(round(rect.left()+expected[0]*rect.width()),round(rect.top()+expected[1]*rect.height()))
    assert canvas.rect().contains(point)
    QTest.mouseClick(canvas,Qt.LeftButton,pos=point)
    assert canvas.points[0]==pytest.approx(expected,abs=.001)
    before=canvas.points.copy()
    canvas.fit_image()
    assert canvas.zoom==1 and canvas.pan==QPointF()
    assert canvas.points==before
    window.close()


def test_wheel_zoom_preserves_the_image_point_under_cursor():
    app,window=editor();canvas=window.canvas
    rect=canvas.area();point=QPointF(rect.center().x()+40,rect.center().y()+20)
    expected=((point.x()-rect.left())/rect.width(),(point.y()-rect.top())/rect.height())
    event=QWheelEvent(point,point,QPoint(),QPoint(0,120),Qt.NoButton,Qt.NoModifier,Qt.NoScrollPhase,False)
    QApplication.sendEvent(canvas,event)
    new=canvas.area()
    assert canvas.zoom>1
    assert ((point.x()-new.left())/new.width(),(point.y()-new.top())/new.height())==pytest.approx(expected)
    window.close()


def test_drag_and_move_mode_do_not_create_annotation_points():
    app,window=editor();canvas=window.canvas;canvas.set_zoom(4)
    center=QPointF(canvas.width()/2,canvas.height()/2)
    for kind,point,button,buttons in [
        (QMouseEvent.MouseButtonPress,center,Qt.LeftButton,Qt.LeftButton),
        (QMouseEvent.MouseMove,center+QPointF(60,40),Qt.NoButton,Qt.LeftButton),
        (QMouseEvent.MouseButtonRelease,center+QPointF(60,40),Qt.LeftButton,Qt.NoButton)]:
        QApplication.sendEvent(canvas,QMouseEvent(kind,point,button,buttons,Qt.NoModifier))
    assert canvas.points==[] and canvas.pan.manhattanLength()>0
    canvas.set_pan_mode(True)
    QTest.mouseClick(canvas,Qt.LeftButton,pos=center.toPoint())
    assert canvas.points==[]
    window.close()

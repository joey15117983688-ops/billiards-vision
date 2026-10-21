import json
import cv2
import numpy as np
import pytest
from PyQt5.QtCore import QPointF,Qt,QEvent
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
import main
from src.exam_editor import ExamDialog,circle_geometry,validate_marks
from src.table_coordinates import TableCoordinates,TableDimensions

@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])


def test_circle_click_resize_delete_and_crop_mapping(app):
    frame=np.zeros((600,1000,3),np.uint8);polygon=[[100,100],[900,100],[900,500],[100,500]]
    dialog=ExamDialog(frame,polygon,TableDimensions(3200,1600,False),[])
    dialog.show();app.processEvents();canvas=dialog.canvas;changes=[];dialog.changed.connect(changes.append)
    position=canvas.mapped(500,300)
    assert np.allclose(canvas.raw(position),[500,300])
    QTest.mouseClick(canvas,Qt.LeftButton,pos=position.toPoint());app.processEvents()
    assert len(changes)==1 and len(canvas.marks)==1
    mark=canvas.marks[0];assert mark['u']==pytest.approx(.5,abs=.002);assert mark['v']==pytest.approx(.5,abs=.002)
    dialog.size_input.setValue(25)
    center,points=circle_geometry(canvas.coordinates,canvas.marks[0])
    assert points[0][0]-center[0]==pytest.approx(50)
    QTest.mouseClick(canvas,Qt.LeftButton,pos=position.toPoint())
    assert len(canvas.marks)==1
    # Outside-table black margins must not create circles.
    QTest.mouseClick(canvas,Qt.LeftButton,pos=QPointF(1,1).toPoint())
    assert len(canvas.marks)==1
    dialog.delete_selected();assert canvas.marks==[]
    QTest.mousePress(canvas,Qt.LeftButton,pos=position.toPoint())
    end=canvas.mapped(560,300)
    QApplication.sendEvent(canvas,QMouseEvent(QEvent.MouseMove,end,Qt.NoButton,Qt.LeftButton,Qt.NoModifier))
    QTest.mouseRelease(canvas,Qt.LeftButton,pos=end.toPoint())
    assert canvas.marks[0]['diameter']==pytest.approx(30,abs=.3)
    dialog.close()


def test_overlay_survives_frames_and_rotates_with_video(app):
    canvas=main.VideoCanvas();canvas.resize(800,600)
    frame=np.zeros((800,400,3),np.uint8)
    polygon=[[20,30],[380,30],[380,770],[20,770]]
    canvas.set_frame(frame);canvas.calibration_polygon=polygon;canvas.recognition_polygon=polygon
    canvas.exam_circles=[{'u':.3,'v':.65,'diameter':12}]
    coordinates=TableCoordinates(polygon,canvas.dimensions)
    center,_=circle_geometry(coordinates,canvas.exam_circles[0])
    canvas.show();app.processEvents()
    before=canvas.grab().toImage()
    canvas.exam_circles=[];without=canvas.grab().toImage();assert before!=without
    canvas.exam_circles=[{'u':.3,'v':.65,'diameter':12}]
    for rotate in [False,True,False]:
        canvas.rotate_portrait=rotate
        point=canvas.display_point(*center);restored=canvas.image_point(point)
        assert np.allclose(restored,center,atol=.01)
        canvas.set_frame(frame);assert len(canvas.exam_circles)==1
        canvas.grab()
    canvas.close()


def test_persistence_and_nonmodal_popup(app,tmp_path,monkeypatch):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    cv2.imwrite(str(tmp_path/'a.png'),np.zeros((600,1000,3),np.uint8))
    window=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    try:
        window.open_exam();assert window.exam_dialog is None
        polygon=[[100,100],[900,100],[900,500],[100,500]]
        window.canvas.calibration_polygon=polygon;window.canvas.recognition_polygon=polygon
        window.open_exam();assert not window.exam_dialog.isModal()
        marks=[{'u':.3,'v':.6,'diameter':20}]
        window.change_exam_marks(marks);window.close_exam()
        path=window.calibration_profile_path().with_suffix('.exam.json')
        assert json.loads(path.read_text())['circles']==marks
        window.canvas.exam_circles=[];window.load_exam_marks();assert window.canvas.exam_circles==marks
        window.open_exam();assert window.exam_dialog.canvas.marks==marks
        window.invalidate_results();assert window.exam_dialog is None and window.canvas.exam_circles==[]
        window.active_calibration_preset=window.calibration_presets.create('different-table')
        window.load_exam_marks();assert window.canvas.exam_circles==[]
    finally:window.close()


@pytest.mark.parametrize('marks',[[{'u':float('nan'),'v':0,'diameter':10}],[{'u':2,'v':0,'diameter':10}],[{'u':0,'v':0,'diameter':0}]])
def test_invalid_saved_marks(marks):
    with pytest.raises(ValueError):validate_marks(marks)


def drag_circle(canvas,start,end):
    QTest.mousePress(canvas,Qt.LeftButton,pos=start.toPoint())
    QApplication.sendEvent(canvas,QMouseEvent(QEvent.MouseMove,end,Qt.NoButton,Qt.LeftButton,Qt.NoModifier))
    QTest.mouseRelease(canvas,Qt.LeftButton,pos=end.toPoint())


@pytest.mark.parametrize('size',[(900,650),(600,800)])
def test_existing_circle_moves_from_grab_point_without_resizing(app,size):
    frame=np.zeros((600,1000,3),np.uint8);polygon=[[100,100],[900,100],[900,500],[100,500]]
    mark={'u':.5,'v':.5,'diameter':30}
    dialog=ExamDialog(frame,polygon,TableDimensions(3200,1600,False),[mark]);dialog.resize(*size)
    dialog.show();app.processEvents();canvas=dialog.canvas;changes=[];dialog.changed.connect(changes.append)
    try:
        # Grab off-centre: the circle must retain its offset instead of snapping.
        drag_circle(canvas,canvas.mapped(510,305),canvas.mapped(610,355))
        assert len(canvas.marks)==1
        moved=canvas.marks[0]
        assert moved['u']==pytest.approx(.625,abs=.003)
        assert moved['v']==pytest.approx(.625,abs=.003)
        assert moved['diameter']==30
        assert changes[-1]==canvas.marks
        assert canvas.anchor is None and canvas.drag is None
        # Dragging past the table clamps the centre to the nearest edge.
        center,_=circle_geometry(canvas.coordinates,moved)
        drag_circle(canvas,canvas.mapped(*center),canvas.mapped(1300,355))
        assert canvas.marks[0]['u']==pytest.approx(1,abs=.001)
        validate_marks(canvas.marks)
    finally:dialog.close()


def test_drag_circle_edge_resizes_and_syncs_diameter_input(app):
    frame=np.zeros((600,1000,3),np.uint8);polygon=[[100,100],[900,100],[900,500],[100,500]]
    mark={'u':.5,'v':.5,'diameter':20}
    dialog=ExamDialog(frame,polygon,TableDimensions(),[mark]);dialog.show();app.processEvents();canvas=dialog.canvas
    try:
        # Click just outside the circumference, within its screen hit tolerance.
        drag_circle(canvas,canvas.mapped(542,300),canvas.mapped(582,300))
        resized=canvas.marks[0]
        assert len(canvas.marks)==1
        assert resized['u']==.5 and resized['v']==.5
        assert resized['diameter']==pytest.approx(40,abs=.3)
        assert dialog.size_input.value()==pytest.approx(resized['diameter'],abs=.01)
        dialog.size_input.setValue(15)
        assert canvas.marks[0]['diameter']==15
    finally:dialog.close()


def test_drag_changes_reach_video_and_persist_on_close(app,tmp_path,monkeypatch):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    cv2.imwrite(str(tmp_path/'a.png'),np.zeros((600,1000,3),np.uint8))
    window=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    try:
        polygon=[[100,100],[900,100],[900,500],[100,500]]
        window.canvas.calibration_polygon=polygon;window.canvas.recognition_polygon=polygon
        window.canvas.exam_circles=[{'u':.5,'v':.5,'diameter':20}]
        window.open_exam();app.processEvents();canvas=window.exam_dialog.canvas
        drag_circle(canvas,canvas.mapped(500,300),canvas.mapped(600,350))
        assert window.canvas.exam_circles==canvas.marks
        assert window.canvas.exam_circles[0]['u']==pytest.approx(.625,abs=.003)
        moved=list(window.canvas.exam_circles)
        window.close_exam();window.load_exam_marks()
        assert window.canvas.exam_circles==moved
        window.open_exam();assert window.exam_dialog.canvas.marks==moved
    finally:window.close()

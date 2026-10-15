import cv2
import numpy as np
import pytest
from PyQt5.QtCore import QPoint, QPointF, QEvent, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication
from src.table_coordinates import TableCoordinates,TableDimensions
from pocket_scene import make_scene
from main import MainWindow


@pytest.mark.parametrize('corners',[
    [[120,180],[1030,70],[1100,650],[180,700]],
    [[200,100],[1000,100],[1000,600],[200,600]],
])
def test_perspective_corners_and_roundtrip_in_mm(corners):
    geometry=TableCoordinates(corners)
    expected=[[0,0],[3569,0],[3569,1778],[0,1778]]
    for point,mm in zip(corners,expected):
        assert geometry.millimetres(*point)==pytest.approx(mm,abs=.002)
    physical=[[300,700],[1784.5,889],[3000,1500]]
    for image,mm in zip(geometry.pixels(physical),physical):
        assert geometry.millimetres(*image)==pytest.approx(mm,abs=.002)
    assert geometry.millimetres(0,0) is None


def test_dimension_validation_and_reversed_corner_order():
    for value in [float('nan'),float('inf'),0,-1,30000]:
        with pytest.raises(ValueError):TableDimensions(value,1778)
    with pytest.raises(ValueError):TableCoordinates([[0,0]]*4)
    geometry=TableCoordinates([[100,100],[100,500],[900,500],[900,100]],TableDimensions(2000,1000,False))
    assert geometry.millimetres(500,300)==pytest.approx([1000,500])


def test_ball_position_uses_box_center():
    geometry=TableCoordinates([[0,0],[1000,0],[1000,500],[0,500]])
    assert geometry.ball_center([100,200,140,240])==pytest.approx([428.28,782.32],abs=.01)
    assert geometry.ball_center([-200,20,-100,40]) is None


@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])


def test_coordinate_hover_selection_and_media_invalidation(app,tmp_path):
    frame,_=make_scene(perspective=True)
    cv2.imwrite(str(tmp_path/'a.jpg'),frame)
    cv2.imwrite(str(tmp_path/'b.jpg'),np.full_like(frame,128))
    window=MainWindow(image_dir=tmp_path,start_with_images=True)
    window.yolo_ready=True  # UI test without a worker process.
    window.show()
    try:
        window.coordinate_button.click();app.processEvents()
        assert window.canvas.coordinates and window.yolo_button.isChecked()
        assert not window.timer.isActive()
        point=window.canvas.coordinates.pixels([[1000,500]])[0]
        area=window.canvas.video_rect()
        q=QPoint(round(area.left()+point[0]*area.width()/frame.shape[1]),round(area.top()+point[1]*area.height()/frame.shape[0]))
        app.sendEvent(window.canvas,QMouseEvent(QEvent.MouseMove,QPointF(q),Qt.NoButton,Qt.NoButton,Qt.NoModifier));app.processEvents()
        assert 'X =' in window.coordinate_readout.text() and 'mm' in window.coordinate_readout.text()
        box=[point[0]-15,point[1]-10,point[0]+15,point[1]+10]
        window.on_yolo_result(window.yolo_generation,'black_ball',.9,box)
        assert window.canvas.coordinate_ball==pytest.approx([1000,500],abs=.01)
        assert '1000.0 mm' in window.ball_coordinate_readout.text()
        window.next_button.click()
        assert window.canvas.coordinates is None and window.canvas.coordinate_ball is None
        assert '未选择' in window.ball_coordinate_readout.text()
        assert '未可靠定位' in window.coordinate_readout.text()
        window.coordinate_button.click()
        assert not window.canvas.coordinate_mode
        assert window.coordinate_readout.text()=='坐标模式未开启'
    finally:window.close()

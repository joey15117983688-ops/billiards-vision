import cv2
import numpy as np
import pytest
from PyQt5.QtCore import QPoint,Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
import main
from main import VideoCanvas,MainWindow

@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])

@pytest.fixture
def window(app,tmp_path,monkeypatch):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    cv2.imwrite(str(tmp_path/'scene.png'),np.zeros((400,200,3),np.uint8))
    w=MainWindow(image_dir=tmp_path,start_with_images=True)
    yield w
    w.close()


def test_clockwise_image_overlay_and_inverse_selection_share_transform(app):
    frame=np.zeros((400,200,3),np.uint8)
    frame[:200,:100]=(0,0,255);frame[:200,100:]=(0,255,0)
    frame[200:,:100]=(255,0,0);frame[200:,100:]=(255,255,255)
    c=VideoCanvas();c.resize(800,450);c.set_frame(frame);c.rotate_portrait=True
    c.selection_enabled=True;clicks=[]
    c.element_clicked.connect(lambda raw,x,y:clicks.append((raw,x,y)))
    assert c.video_rect().width()>c.video_rect().height()
    for raw in [(30,60),(150,60),(30,320),(150,320)]:
        assert c.image_point(c.display_point(*raw))==pytest.approx(raw)
    c.show();app.processEvents()
    shot=c.grab().toImage()
    # Top-left red source quadrant moves to the screen's top-right quadrant.
    point=c.display_point(40,70)
    assert shot.pixelColor(round(point.x()),round(point.y())).red()>200
    raw=(40,300);point=c.display_point(*raw)
    QTest.mouseClick(c,Qt.LeftButton,pos=QPoint(round(point.x()),round(point.y())))
    assert clicks[-1][1:]==pytest.approx(raw,abs=1)
    assert np.array_equal(clicks[-1][0],frame)
    polygon=c.canvas_polygon([[30,60],[60,60],[60,110],[30,110]])
    assert polygon.boundingRect()==c.box_rect([30,60,60,110])
    c.close()


def test_fullscreen_button_rotates_portrait_preserves_geometry_and_escape(window,app):
    w=window;w.show();app.processEvents()
    polygon=[[20,20],[180,20],[180,380],[20,380]]
    w.canvas.recognition_polygon=polygon
    before=w.canvas.frame.copy()
    geometry=w.geometry();canvas_size=w.canvas.size()
    w.canvas.fullscreen_button.click();app.processEvents()
    assert not w.isFullScreen() and w.canvas.is_rotated()
    assert w.geometry()==geometry and w.canvas.size()==canvas_size
    assert all(not widget.isHidden() for widget in w.fullscreen_controls)
    assert np.array_equal(w.canvas.frame,before) and w.canvas.recognition_polygon==polygon
    assert w.canvas.fullscreen_button.x()+w.canvas.fullscreen_button.width()<=w.canvas.width()
    QTest.keyClick(w,Qt.Key_Escape);app.processEvents()
    assert not w.video_fullscreen and not w.canvas.is_rotated()
    assert not w.fullscreen_controls[0].isHidden()
    w.canvas.set_frame(np.zeros((200,400,3),np.uint8))
    w.canvas.fullscreen_button.click();app.processEvents()
    assert w.video_fullscreen and not w.canvas.is_rotated()
    w.canvas.fullscreen_button.click()


def event(shot,ball,status):return {'shot_id':shot,'events':[{'ball':ball,'event':status}]}


def test_count_confirmed_pots_deduplicates_and_retracts(window):
    w=window
    assert w.pot_count_label.text()=='（0）'
    w.update_pot_count([event('shot_0001','black_ball','POT_CANDIDATE')])
    assert w.pot_count_label.text()=='（0）'
    confirmed=event('shot_0001','black_ball','POTTED')
    w.update_pot_count([confirmed]);w.update_pot_count([confirmed])
    assert w.pot_count_label.text()=='（1）'
    w.update_pot_count([event('shot_0001','white_ball','POTTED')])
    assert w.pot_count_label.text()=='（2）'
    w.update_pot_count([event('shot_0001','black_ball','NOT_POTTED')])
    assert w.pot_count_label.text()=='（1）'
    w.reset_pot_session('关闭观察')
    assert w.pot_count_label.text()=='（1）'
    w.update_pot_count([confirmed])
    assert w.pot_count_label.text()=='（2）'
    w.show_image(0)
    assert w.pot_count_label.text()=='（0）'

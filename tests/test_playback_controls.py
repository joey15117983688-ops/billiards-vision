"""Manual start and one playback toggle, including asynchronous selection."""
import cv2
import numpy as np
import pytest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication,QPushButton
import main

@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])

@pytest.fixture
def window(app,tmp_path,monkeypatch):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    path=tmp_path/'test.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,100))
    assert writer.isOpened()
    for n in range(100):writer.write(np.full((100,160,3),(30+n,120,30),np.uint8))
    writer.release();cv2.imwrite(str(tmp_path/'scene.png'),np.zeros((100,160,3),np.uint8))
    w=main.MainWindow(path,image_dir=tmp_path)
    yield w
    w.close()


def test_startup_stays_on_first_frame_until_single_button_clicked(window):
    w=window
    assert w.playback_ready and not w.timer.isActive()
    assert w.play_button.text()=='播放' and w.play_button.isEnabled()
    assert [b for b in w.findChildren(QPushButton) if b.text() in ('播放','暂停')]==[w.play_button]
    position=w.capture.get(cv2.CAP_PROP_POS_FRAMES)
    QTest.qWait(160)
    assert w.capture.get(cv2.CAP_PROP_POS_FRAMES)==position
    w.play_button.click()
    assert w.timer.isActive() and w.play_button.text()=='暂停' and w.play_button.isEnabled()
    QTest.qWait(140)
    assert w.capture.get(cv2.CAP_PROP_POS_FRAMES)>position
    w.play_button.click()
    assert not w.timer.isActive() and w.play_button.text()=='播放'
    paused=w.capture.get(cv2.CAP_PROP_POS_FRAMES)
    QTest.qWait(150)
    assert w.capture.get(cv2.CAP_PROP_POS_FRAMES)==paused
    w.play_button.click()
    assert w.timer.isActive() and w.play_button.text()=='暂停'
    assert w.capture.get(cv2.CAP_PROP_POS_FRAMES)==paused


def test_source_switch_and_eof_leave_toggle_ready_to_play(window):
    w=window;w.play_button.click()
    w.source_combo.setCurrentIndex(1)
    assert not w.timer.isActive() and not w.play_button.isEnabled()
    w.source_combo.setCurrentIndex(0)
    assert not w.timer.isActive() and w.play_button.text()=='播放' and w.play_button.isEnabled()
    w.play_button.click();w.capture.set(cv2.CAP_PROP_POS_FRAMES,100)
    w.next_frame()
    assert not w.timer.isActive() and w.play_button.text()=='播放'
    assert w.play_button.isEnabled() and w.capture.get(cv2.CAP_PROP_POS_FRAMES)==1


def test_pending_play_request_can_be_cancelled_with_same_button(window):
    w=window;w.selection_busy=True
    w.play_button.click()
    assert w.resume_after_selection and not w.timer.isActive() and w.play_button.text()=='暂停'
    w.play_button.click()
    assert not w.resume_after_selection and not w.timer.isActive() and w.play_button.text()=='播放'
    w.selection_busy=False
    w.play_button.click();assert w.timer.isActive()


def test_failed_selection_restores_play_label_and_calibration_disables_toggle(window):
    w=window;w.yolo_ready=True;w.yolo_button.setChecked(True);w.selection_busy=True
    w.play_button.click();assert w.play_button.text()=='暂停'
    w.on_yolo_result(w.yolo_generation,None,0)
    assert not w.timer.isActive() and w.play_button.text()=='播放'
    w.calibration_controls(True);assert not w.play_button.isEnabled()
    w.calibration_controls(False);assert w.play_button.isEnabled()

"""Independent pocket annotations must not leak across media or disappear with YOLO toggles."""
import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from main import MainWindow
from pocket_scene import make_scene
from src.vision.pocket import PocketDetector


@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])


def test_pocket_result_survives_yolo_toggle_but_not_image_change(app,tmp_path):
    frame,_=make_scene()
    cv2.imwrite(str(tmp_path/'a.jpg'),frame)
    cv2.imwrite(str(tmp_path/'b.jpg'),frame)
    window=MainWindow(tmp_path/'missing.mp4',image_dir=tmp_path,start_with_images=True)
    try:
        result=PocketDetector().detect(frame).to_dict()
        generation=window.media_generation
        window.on_pocket_result(generation,result)
        assert window.canvas.pocket_result['valid']
        window.yolo_ready=True
        window.yolo_button.setChecked(True)
        window.yolo_button.setChecked(False)
        assert window.canvas.pocket_result is result
        window.next_button.click()
        assert window.canvas.pocket_result is None
        window.on_pocket_result(generation,result)
        assert window.canvas.pocket_result is None
    finally:window.close()


def test_pocket_error_restores_button_and_does_not_crash(app,tmp_path):
    cv2.imwrite(str(tmp_path/'a.jpg'),np.zeros((240,320,3),np.uint8))
    window=MainWindow(image_dir=tmp_path,start_with_images=True)
    try:
        window.pocket_busy=True;window.update_pocket_button()
        assert not window.pocket_button.isEnabled()
        window.on_pocket_error(window.media_generation,'test disk failure')
        assert window.pocket_button.isEnabled() and not window.pocket_busy
        assert 'test disk failure' in window.statusBar().currentMessage()
    finally:window.close()

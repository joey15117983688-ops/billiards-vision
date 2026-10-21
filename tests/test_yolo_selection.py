"""Verify click-to-frame mapping and ignoring results after YOLO is disabled."""
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from main import MainWindow, VideoCanvas


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_click_mapping_and_non_video_regions(app):
    canvas = VideoCanvas()
    canvas.resize(640, 480)
    canvas.set_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
    canvas.selection_enabled = True
    clicks = []
    canvas.element_clicked.connect(lambda frame, x, y: clicks.append((x, y)))
    canvas.show()
    app.processEvents()
    QTest.mouseClick(canvas, Qt.LeftButton, pos=canvas.video_rect().center().toPoint())
    assert len(clicks) == 1
    assert clicks[0][0] == pytest.approx(960, abs=3)
    assert clicks[0][1] == pytest.approx(540, abs=3)
    QTest.mouseClick(canvas, Qt.LeftButton, pos=QPoint(10, 10))  # Letterbox bar.
    QTest.mouseClick(canvas, Qt.LeftButton, pos=QPoint(544, 96))
    assert len(clicks) == 2  # Former badge area now belongs to the video.
    canvas.selection_enabled = False
    QTest.mouseClick(canvas, Qt.LeftButton, pos=canvas.video_rect().center().toPoint())
    assert len(clicks) == 2
    canvas.close()


@pytest.mark.parametrize("name,label", [("billiards_table", "台球桌"), ("white_ball", "白球")])
def test_results_do_not_reappear_when_disabled(app, tmp_path, name, label):
    window = MainWindow(tmp_path / "missing.mp4")
    try:
        window.on_yolo_result(window.yolo_generation, name, 0.9)
        assert window.canvas.selected_box is None
        window.yolo_ready = True  # Exercise display without starting a model process.
        window.yolo_button.setChecked(True)
        generation = window.yolo_generation
        window.on_yolo_result(generation, name, 0.9, [10, 20, 30, 40])
        assert window.canvas.selected_label == f"{label} 90%"
        window.yolo_button.setChecked(False)
        window.on_yolo_result(generation, name, 0.9, [10, 20, 30, 40])
        assert window.canvas.selected_box is None
        window.yolo_button.setChecked(True)
        window.on_yolo_result(generation, name, 0.9, [10, 20, 30, 40])
        assert window.canvas.selected_box is None
    finally:
        window.close()


def test_selected_box_scales_with_letterboxed_video(app):
    canvas = VideoCanvas()
    canvas.resize(640, 480)
    canvas.set_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
    canvas.set_selection([480, 270, 1440, 810], "台球 90%")
    rect = canvas.selected_rect()
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == pytest.approx((160, 150, 320, 180))
    canvas.resize(1280, 720)
    rect = canvas.selected_rect()
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == pytest.approx((320, 180, 640, 360))
    canvas.set_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert canvas.selected_box is None
    canvas.close()


def test_identify_all_and_discard_results_after_play_or_disable(app, tmp_path):
    window = MainWindow(tmp_path / 'missing.mp4')
    requests = []
    try:
        window.playback_ready = True
        window.canvas.set_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
        window.yolo_ready = True
        window.identify_all_requested.connect(lambda frame, gen: requests.append((frame, gen)))
        window.update_all_button()
        window.identify_all_button.click()
        assert window.yolo_button.isChecked() and not window.timer.isActive()
        assert len(requests) == 1 and window.identify_all_button.isEnabled()
        generation = requests[0][1]
        detections = [
            {'name': 'billiards_table', 'confidence': .9, 'box': [0, 0, 1900, 1000]},
            {'name': 'white_ball', 'confidence': .9, 'box': [10, 20, 30, 40]},
        ]
        window.on_yolo_all_result(generation, detections)
        assert len(window.canvas.detections) == 2
        assert '白球 1 个' in window.statusBar().currentMessage()
        assert window.identify_all_button.isEnabled()
        window.identify_all_button.click()
        generation = requests[-1][1]
        window.play_video()
        window.pause_video()
        window.on_yolo_all_result(generation, detections)
        assert window.canvas.detections == []
        window.identify_all_button.click()
        generation = requests[-1][1]
        window.yolo_button.setChecked(False)
        window.on_yolo_all_result(generation, detections)
        assert window.canvas.detections == []
    finally:
        window.close()


def test_pending_all_cancels_when_playback_resumes(app, tmp_path):
    window = MainWindow(tmp_path / 'missing.mp4')
    try:
        window.playback_ready = True
        window.yolo_loading = True
        window.canvas.set_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
        window.identify_all_elements()
        assert window.pending_all is not None
        window.play_video()
        window.pause_video()
        assert window.pending_all is None
        window.on_yolo_ready()
        assert not window.yolo_busy and window.canvas.detections == []
    finally:
        window.close()


def test_gallery_switch_discards_previous_image_result(app, tmp_path):
    import cv2
    cv2.imwrite(str(tmp_path/'a.jpg'), np.full((160, 240, 3), (40, 120, 40), np.uint8))
    cv2.imwrite(str(tmp_path/'b.jpg'), np.full((160, 240, 3), (160, 50, 50), np.uint8))
    window = MainWindow(tmp_path/'missing.mp4', image_dir=tmp_path, start_with_images=True)
    try:
        assert window.capture is None
        assert window.image_index == 0 and '1 / 2' in window.image_info.text()
        assert not window.previous_button.isEnabled() and window.next_button.isEnabled()
        assert not window.play_button.isEnabled() and not window.timer.isActive()
        window.yolo_ready = True
        window.identify_all_button.click()
        generation = window.yolo_generation
        window.next_button.click()
        assert window.image_index == 1 and not window.next_button.isEnabled()
        assert window.canvas.frame[0, 0, 0] > 120
        window.on_yolo_all_result(generation, [{'name':'white_ball','confidence':.9,'box':[1,1,20,20]}])
        assert window.canvas.detections == []
        window.previous_button.click()
        assert window.image_index == 0
    finally:
        window.close()


def test_gallery_can_leave_undecodable_image(app, tmp_path):
    import cv2
    (tmp_path/'a.jpg').write_bytes(b'not an image')
    cv2.imwrite(str(tmp_path/'b.jpg'), np.zeros((100, 200, 3), np.uint8))
    window = MainWindow(tmp_path/'missing.mp4', image_dir=tmp_path, start_with_images=True)
    try:
        assert not window.identify_all_button.isEnabled()
        assert window.next_button.isEnabled()
        window.next_button.click()
        assert window.playback_ready and window.identify_all_button.isEnabled()
    finally:
        window.close()

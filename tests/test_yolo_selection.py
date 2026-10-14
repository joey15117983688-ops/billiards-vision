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
    QTest.mouseClick(canvas, Qt.LeftButton, pos=canvas.badge_rect().center().toPoint())
    assert len(clicks) == 1
    canvas.selection_enabled = False
    QTest.mouseClick(canvas, Qt.LeftButton, pos=canvas.video_rect().center().toPoint())
    assert len(clicks) == 1
    canvas.close()


def test_results_do_not_reappear_when_disabled(app, tmp_path):
    window = MainWindow(tmp_path / "missing.mp4")
    try:
        window.on_yolo_result(window.yolo_generation, "billiards_table", 0.9)
        assert window.canvas.badge_text == "1"
        window.yolo_ready = True  # Exercise display without starting a model process.
        window.yolo_button.setChecked(True)
        generation = window.yolo_generation
        window.on_yolo_result(generation, "billiards_table", 0.9)
        assert window.canvas.badge_text == "台球桌"
        window.yolo_button.setChecked(False)
        window.on_yolo_result(generation, "billiards_table", 0.9)
        assert window.canvas.badge_text == "1"
        window.yolo_button.setChecked(True)
        window.on_yolo_result(generation, "billiards_table", 0.9)
        assert window.canvas.badge_text == "请点击\n物体"
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

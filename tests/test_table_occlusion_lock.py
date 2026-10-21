"""Validated table geometry must survive players crossing the cloth and rails."""
import copy
from pathlib import Path
import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from main import VideoCanvas
from src.cloth_region import cloth_region
from src.target_balls import scene_detections
from scripts.yolo_click_worker import detect_selected
from pocket_scene import make_scene

@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])


def test_calibrated_canvas_rejects_transient_table_boxes_but_allows_explicit_recalibration(app):
    canvas=VideoCanvas();frame,_=make_scene();canvas.set_frame(frame)
    inner=[[200,200],[1000,200],[1000,600],[200,600]]
    scope=[[185,185],[1015,185],[1015,615],[185,615]]
    canvas.calibration_polygon=inner;canvas.recognition_polygon=scope
    canvas.pocket_result={'table':{'playable_polygon':inner,'scope_polygon':copy.deepcopy(scope)},'pockets':[]}
    canvas.coordinate_mode=True;canvas.calibrate_coordinates()
    before=canvas.coordinates.millimetres(500,400)
    for offset in (40,-55,100,0):
        shifted=[[x+offset,y] for x,y in scope]
        raw=[{'name':'billiards_table','polygon':shifted,'box':[0,0,1280,800]}]
        canvas.set_detections(raw)
        assert canvas.table_outline()==scope
        assert canvas.detections[0]['polygon']==scope
        assert canvas.pocket_result['table']['scope_polygon']==scope
        assert canvas.coordinates.millimetres(500,400)==pytest.approx(before)
        assert raw[0]['polygon']==shifted
    corrected=[[x+10,y+5] for x,y in scope]
    canvas.recognition_polygon=corrected
    canvas.set_detections([{'name':'billiards_table','polygon':scope}])
    assert canvas.table_outline()==corrected
    canvas.close()


def test_worker_reuses_scope_under_arm_and_same_colour_bystander(monkeypatch):
    frame,_=make_scene();inner=[[200,200],[1000,200],[1000,600],[200,600]]
    scope=cloth_region(frame,inner)['polygon']
    import src.cloth_region
    monkeypatch.setattr(src.cloth_region,'cloth_region',lambda *a:pytest.fail('validated scope must not be refitted every frame'))
    for x in (100,190,230,600,1000,1150):
        occluded=frame.copy()
        cv2.rectangle(occluded,(x-35,100),(x+35,450),(100,150,210),-1)
        # A moving green shirt outside the table must not enlarge the locked scope.
        cv2.rectangle(occluded,(0,250),(190,500),(45,145,45),-1)
        for table in (scene_detections(occluded,[],inner,scope)[0],
                      detect_selected(None,occluded,{'name':'billiards_table'},inner,scope)[0]):
            assert table['polygon']==scope and table['playable_polygon']==inner


def test_original_video_table_scope_does_not_move_during_player_occlusion():
    source=Path(__file__).resolve().parents[1]/'data/test.mp4'
    if not source.exists():pytest.skip('Local original recording unavailable')
    capture=cv2.VideoCapture(str(source))
    try:
        ok,frame=capture.read();assert ok
        inner=[[138,149],[929,149],[936,1762],[143,1762]]
        scope=cloth_region(frame,inner)['polygon']
        for second in (1,5,38,40,47,65):
            capture.set(cv2.CAP_PROP_POS_MSEC,second*1000)
            ok,frame=capture.read();assert ok
            result=scene_detections(frame,[],inner,scope)[0]
            assert result['polygon']==scope
            assert result['playable_polygon']==inner
    finally:capture.release()

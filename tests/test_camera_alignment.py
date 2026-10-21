"""Camera motion must move annotations, without rewriting the calibration."""
from copy import deepcopy
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
import main
from src.camera_alignment import CameraAlignment,encode_reference,map_points,transform_geometry
from src.cloth_region import cloth_region
from src.calibration_state import save_automatic
from src.exam_editor import circle_geometry
from src.table_coordinates import TableCoordinates
from src.vision.pocket import PocketDetector,TableResult
from src.vision.pocket_event.adapter import PocketEventAdapter

ROOT=Path(main.__file__).parent


def video_frame(seconds):
    cap=cv2.VideoCapture(str(ROOT/'data/test.mp4'))
    cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000);ok,frame=cap.read();cap.release()
    assert ok
    return frame


def scene(frame):
    inner=[[138,149],[929,149],[936,1762],[143,1762]]
    scope=cloth_region(frame,inner)['polygon']
    table=TableResult(detected=True,complete=True,corners=inner,playable_polygon=inner,scope_polygon=scope,quality={'geometry_reliable':True})
    pockets=PocketDetector().detect(frame,table=table).to_dict()
    return inner,scope,pockets


def test_real_video_zoom_moves_outline_and_holds_when_players_move():
    first=video_frame(0);inner,scope,pockets=scene(first)
    alignment=CameraAlignment(first,scope)
    moved=video_frame(30);matrix=alignment.update(moved,force=True)
    assert matrix is not None and not alignment.lost
    actual=cloth_region(moved,inner)['polygon']
    assert np.max(np.linalg.norm(np.asarray(map_points(scope,matrix))-actual,axis=1))<3
    assert np.linalg.norm(matrix-np.eye(3))>1
    old=matrix.copy()
    # Many ball and player changes, same camera framing.
    for t in [40,50,67,90]:
        alignment.update(video_frame(t),force=True)
        assert not alignment.lost
        assert np.max(np.linalg.norm(np.asarray(map_points(scope,alignment.matrix))-np.asarray(map_points(scope,old)),axis=1))<3
    alignment.update(np.zeros_like(first),force=True)
    assert alignment.lost and np.array_equal(alignment.matrix,old)


def test_transformed_pockets_and_exam_center_share_one_mapping():
    frame=video_frame(0);polygon,scope,pockets=scene(frame);original=deepcopy(pockets)
    matrix=np.array([[.965,0,18],[0,.965,28],[0,0,1.]])
    inner,moved_scope,moved=transform_geometry(polygon,scope,pockets,matrix)
    assert pockets==original
    for old,new in zip(pockets['pockets'],moved['pockets']):
        p=old.get('center') or old['candidate_center'];q=new.get('center') or new['candidate_center']
        assert np.allclose([q['x'],q['y']],map_points([[p['x'],p['y']]],matrix)[0])
    mark={'u':.3,'v':.6,'diameter':20}
    c1,points1=circle_geometry(TableCoordinates(scope),mark)
    c2,points2=circle_geometry(TableCoordinates(moved_scope),mark)
    assert np.allclose(c2,map_points([c1],matrix)[0],atol=.001)
    assert np.allclose(points2,map_points(points1,matrix),atol=.001)
    adapter=PocketEventAdapter(pockets);engine=adapter.engine
    adapter.align_geometry(moved,matrix)
    assert adapter.engine is engine
    for old,new in zip(pockets['pockets'],moved['pockets']):
        p=old.get('center') or old['candidate_center'];q=new.get('center') or new['candidate_center']
        before=PocketEventAdapter(pockets).project([[p['x'],p['y']]])
        assert np.allclose(before,adapter.project([[q['x'],q['y']]]),atol=1e-6)


def test_saved_reference_registers_after_restart_at_another_video_time(tmp_path):
    frame=video_frame(0);polygon,scope,pockets=scene(frame)
    path=tmp_path/'camera.auto.json'
    save_automatic(path,frame,{'polygon':polygon,'scope_polygon':scope,'pockets':pockets,'calibration_report':{}},None)
    original=path.read_bytes();data=json.loads(original)
    current=video_frame(30)
    restored=CameraAlignment(current,scope,data['alignment_reference'])
    restored.update(current,force=True)
    assert not restored.lost
    measured=cloth_region(current,polygon)['polygon']
    assert np.max(np.linalg.norm(np.asarray(map_points(scope,restored.matrix))-measured,axis=1))<3
    assert path.read_bytes()==original


def test_ui_aligns_on_current_frames_including_without_yolo(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(ROOT/'data/test.mp4')
    try:
        w.pause_video();frame=video_frame(0);inner,scope,pockets=scene(frame)
        w.canvas.set_frame(frame);w.canvas.calibration_polygon=inner;w.canvas.recognition_polygon=scope;w.canvas.set_pocket_result(pockets)
        w.canvas.exam_circles=[{'u':.3,'v':.6,'diameter':20}]
        w.start_camera_alignment()
        baseline=deepcopy(w.camera_geometry)
        w.yolo_worker=main.YoloWorker()
        w.capture.set(cv2.CAP_PROP_POS_MSEC,30000)
        for _ in range(4):w.next_frame()
        measured=cloth_region(w.canvas.frame,inner)['polygon']
        assert np.max(np.linalg.norm(np.asarray(w.canvas.recognition_polygon)-measured,axis=1))<3
        assert w.camera_geometry==baseline
        assert w.yolo_worker.scope_polygon==w.canvas.recognition_polygon
        assert w.last_pocket_result==w.canvas.pocket_result
        center,_=circle_geometry(TableCoordinates(w.canvas.recognition_polygon),w.canvas.exam_circles[0])
        for rotated in [True,False]:
            w.canvas.rotate_portrait=rotated
            assert np.allclose(w.canvas.image_point(w.canvas.display_point(*center)),center,atol=.01)
        assert not list((tmp_path/'config').rglob('*.auto.json'))
        w.invalidate_results();assert w.camera_alignment is None and w.camera_geometry is None
    finally:w.close()


def test_textureless_or_local_occlusion_cannot_move_table():
    frame=video_frame(0);_,scope,_=scene(frame)
    alignment=CameraAlignment(frame,scope)
    local=frame.copy();cv2.rectangle(local,(0,800),(300,1400),(110,150,210),-1)
    alignment.update(local,force=True)
    assert not alignment.lost and np.array_equal(alignment.matrix,np.eye(3))
    covered=frame.copy();cv2.rectangle(covered,(0,0),(1080,1922),(60,120,180),-1)
    alignment.update(covered,force=True)
    assert alignment.lost and np.array_equal(alignment.matrix,np.eye(3))

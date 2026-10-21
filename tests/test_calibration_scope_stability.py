"""Regressions for the Oct 19 report: skewed outline and changing retained results."""
from copy import deepcopy
import json
from pathlib import Path
import cv2
import numpy as np
from src.cloth_region import cloth_region,scope_improvement
from src.scene_calibration import calibrate_scene
from src.calibration_state import save_automatic

ROOT=Path(__file__).resolve().parents[1]


def test_real_portrait_frame_rejects_slanted_outer_rail():
    cap=cv2.VideoCapture(str(ROOT/'data/test.mp4'))
    cap.set(cv2.CAP_PROP_POS_MSEC,67000);ok,frame=cap.read();cap.release()
    assert ok
    inner=[[138,149],[929,149],[936,1762],[143,1762]]
    wrong=[[93.57,111.88],[964.47,107.86],[990.68,1809.32],[105.56,1848.61]]
    region=cloth_region(frame,inner);q=np.array(region['polygon'])
    assert abs(q[2,1]-q[3,1])<5  # Reported long edge must follow the visible rail.
    assert abs(q[1,0]-q[2,0])<8
    assert 110<q[0,0]<125 and 130<q[0,1]<150
    assert 1760<q[3,1]<1790
    assert scope_improvement(frame,wrong,region['polygon'],region['hue'])


def test_body_and_same_hue_protrusion_do_not_pull_boundary():
    frame=np.zeros((600,1000,3),np.uint8)
    cv2.rectangle(frame,(100,100),(900,500),(0,160,0),-1)
    # Same-colour attachment near the bottom left defeats hull/percentile fitting.
    cv2.rectangle(frame,(100,455),(235,535),(0,160,0),-1)
    cv2.rectangle(frame,(80,235),(180,290),(110,150,210),-1)
    inner=[[125,125],[875,125],[875,475],[125,475]]
    result=cloth_region(frame,inner);q=np.array(result['polygon'])
    assert np.allclose(q,[[100,100],[900,100],[900,500],[100,500]],atol=5)


def test_unchanged_preserves_precise_scope_and_pocket_centres(monkeypatch):
    import src.adaptive_calibration as adaptive
    from src.vision.pocket import TableResult,PocketDetector
    frame=np.zeros((600,1000,3),np.uint8);cv2.rectangle(frame,(100,100),(900,500),(0,160,0),-1)
    polygon=[[125,125],[875,125],[875,475],[125,475]]
    scope=[[100.2,100.1],[899.8,100.3],[899.9,499.7],[100.1,499.8]]
    table=TableResult(detected=True,complete=True,corners=polygon,playable_polygon=polygon,scope_polygon=scope,quality={})
    pockets=PocketDetector().detect(frame,table=table).to_dict()
    pockets['pockets'][0]['center']={'x':104.321,'y':103.456}
    original=deepcopy(pockets)
    monkeypatch.setattr(adaptive,'search_calibration',lambda *args:(polygon,deepcopy(pockets),{'decision':'unchanged'}))
    result=calibrate_scene(frame,None,lambda _:None,correction_context={'previous_scope':scope,'previous_pockets':pockets},geometry_only=True)
    assert result['scope_polygon']==scope
    assert result['pockets']==original and pockets==original
    assert result['calibration_report']['scope_corrected'] is False


def test_saves_have_timestamp_previous_version_and_noop_preserves_file(tmp_path):
    from src.vision.pocket import TableResult,PocketDetector
    frame=np.zeros((600,1000,3),np.uint8);polygon=[[100,100],[900,100],[900,500],[100,500]]
    table=TableResult(detected=True,complete=True,corners=polygon,playable_polygon=polygon,scope_polygon=polygon,quality={})
    pockets=PocketDetector().detect(frame,table=table).to_dict()
    result={'polygon':polygon,'pockets':pockets,'calibration_report':{'decision':'initial'}}
    path=tmp_path/'camera.auto.json';save_automatic(path,frame,result,None)
    original=path.read_bytes();assert json.loads(original)['saved_at']
    save_automatic(path,frame,result,None);assert path.read_bytes()==original
    result['scope_polygon']=[[95,95],[905,95],[905,505],[95,505]]
    save_automatic(path,frame,result,None)
    assert path.with_suffix('.previous.json').read_bytes()==original
    assert json.loads(path.read_text())['pockets']['table']['scope_polygon']==result['scope_polygon']

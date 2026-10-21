from pathlib import Path
import json
import cv2
import numpy as np
import pytest
from src.adaptive_calibration import search_calibration,choose_correction,same_polygon
from src.calibration_state import begin_feedback,save_automatic,load_automatic

FIXTURE=Path(__file__).parent/'fixtures/calibration_portrait.png'
WRONG=[[40,62],[638,60],[641,1159],[36,1159]]


def test_reported_black_rails_are_removed_after_comparing_alternatives():
    frame=cv2.imread(str(FIXTURE))
    polygon,pockets,report=search_calibration(frame,lambda _:None,
        {'attempt':1,'previous_polygon':WRONG,'rejected_polygons':[WRONG]})
    assert report['decision']=='updated' and report['candidate_count']>20
    assert report['before']['black_band']>.4 and report['after']['black_band']<.05
    assert report['after']['side_observed']==2 and pockets['valid']
    assert np.all(np.array(polygon)[[0,3],0]>78)
    assert np.all(np.array(polygon)[[1,2],0]<615)
    assert not same_polygon(WRONG,polygon)


def test_missing_both_sides_is_repaired_by_outward_search():
    frame=cv2.imread(str(FIXTURE));too_small=[[140,84],[560,84],[564,1136],[143,1136]]
    polygon,pockets,report=search_calibration(frame,lambda _:None,
        {'attempt':1,'previous_polygon':too_small,'rejected_polygons':[too_small]})
    assert report['decision']=='updated' and report['after']['side_observed']==2
    assert polygon[0][0]<100 and polygon[1][0]>590


def test_repeated_feedback_expands_search_and_does_not_degrade_good_result():
    frame=cv2.imread(str(FIXTURE))
    polygon,_,_=search_calibration(frame,lambda _:None)
    _,_,first=search_calibration(frame,lambda _:None,{'attempt':1,'previous_polygon':polygon,'rejected_polygons':[polygon]})
    kept,_,third=search_calibration(frame,lambda _:None,{'attempt':3,'previous_polygon':polygon,'rejected_polygons':[polygon]})
    assert third['search_radius_ratio']>first['search_radius_ratio']
    assert {e['strategy'] for e in first['evaluations']}!={e['strategy'] for e in third['evaluations']}
    assert first['decision']==third['decision']=='unchanged'
    assert same_polygon(kept,polygon)


def test_known_rejected_result_cannot_cycle_back_and_bad_replacement_is_refused():
    def candidate(polygon,score,observed=6):
        return {'polygon':polygon,'metrics':{'reliable':True,'score':score,'observed':observed,'side_observed':2}}
    base=candidate(WRONG,10)
    rejected=[[80,84],[600,84],[604,1136],[84,1136]]
    good=[[90,84],[590,84],[594,1136],[94,1136]]
    result,decision=choose_correction([candidate(rejected,20),candidate(good,12)],base,[WRONG,rejected])
    assert decision=='updated' and result['polygon']==good
    held,decision=choose_correction([candidate(good,20,observed=4)],base,[WRONG])
    assert decision=='unchanged' and held is base
    held,decision=choose_correction([candidate(good,10.8)],base,[WRONG],manual=True)
    assert decision=='unchanged' and held is base


def test_feedback_and_accepted_corrections_survive_restart_but_reject_changed_scene(tmp_path):
    frame=cv2.imread(str(FIXTURE));path=tmp_path/'source.feedback.json'
    first=begin_feedback(path,frame,WRONG,None)
    second=begin_feedback(path,frame,WRONG,None)
    assert first['attempt']==1 and second['attempt']==2
    assert len(json.loads(path.read_text())['rejected'])==1
    polygon,pockets,report=search_calibration(frame,lambda _:None,second)
    result={'polygon':polygon,'pockets':pockets,'calibration_report':report}
    auto=tmp_path/'source.auto.json';save_automatic(auto,frame,result,None)
    restored=load_automatic(auto,frame,None)
    assert same_polygon(restored[0],polygon) and restored[1]['valid']
    assert load_automatic(auto,np.zeros_like(frame),None) is None
    assert load_automatic(auto,frame,{'new':'human annotation'}) is None
    changed=begin_feedback(path,frame,polygon,{'new':'human annotation'})
    assert changed['attempt']==1


def test_exposure_change_can_fix_black_rails_without_faking_verified_pockets():
    frame=cv2.imread(str(FIXTURE));frame=np.clip(frame.astype(float)*1.5,0,255).astype(np.uint8)
    polygon,pockets,report=search_calibration(frame,lambda _:None,
        {'attempt':2,'previous_polygon':WRONG,'rejected_polygons':[WRONG]})
    assert report['decision']=='updated'
    assert report['after']['black_band']<.06
    assert np.all(np.array(polygon)[[0,3],0]>78)
    assert np.all(np.array(polygon)[[1,2],0]<615)
    assert not pockets['valid']
    assert sum(p['status']=='observed' for p in pockets['pockets'])<6


def test_saved_pocket_geometry_scales_consistently_with_resolution(tmp_path):
    frame=cv2.imread(str(FIXTURE));polygon,pockets,report=search_calibration(frame,lambda _:None)
    path=tmp_path/'saved.auto.json'
    save_automatic(path,frame,{'polygon':polygon,'pockets':pockets,'calibration_report':report},None)
    large=cv2.resize(frame,None,fx=2,fy=2)
    restored,data=load_automatic(path,large,None)
    assert np.allclose(restored,np.asarray(polygon)*2)
    geometry=data['geometry'];q=np.float32([geometry['ordered_corners']])
    mapped=cv2.perspectiveTransform(q,np.asarray(geometry['image_to_normalized']))[0]
    assert np.allclose(mapped,[[0,0],[1,0],[1,1],[0,1]],atol=1e-5)

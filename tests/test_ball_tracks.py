import json
import cv2
import numpy as np
from src.ball_tracks import BallTracks
from src.track_recording import TrackRecorder
from src.table_coordinates import TableCoordinates,TableDimensions


def frame_with(red,blue=None):
    frame=np.full((100,180,3),(45,140,45),np.uint8)
    cv2.circle(frame,tuple(red),8,(20,20,220),-1)
    if blue is not None:cv2.circle(frame,tuple(blue),8,(220,30,20),-1)
    return frame


def item(point,selection=None):
    x,y=point;d={'name':'billiard_ball','box':[x-8,y-8,x+9,y+9],'confidence':.9}
    if selection is not None:d['selection_id']=selection
    return d


def test_same_class_crossing_reordered_detections_keeps_identity():
    tracker=BallTracks();red_id=blue_id=None
    for i,(r,b) in enumerate([((30,50),(130,50)),((45,50),(115,50)),((60,50),(100,50)),((75,50),(85,50)),((90,50),(70,50)),((105,50),(55,50))]):
        detections=[item(r),item(b)]
        if i%2:detections.reverse()
        found=tracker.update(i,frame_with(r,b),detections)
        if i==0:red_id,blue_id=[d['track_id'] for d in found]
        if i in (4,5):
            assert {d['track_id'] for d in found}=={red_id,blue_id}
            assert np.linalg.norm(tracker.states[red_id]['center']-r)<2
            assert np.linalg.norm(tracker.states[blue_id]['center']-b)<2


def test_short_occlusion_recovers_same_id_and_breaks_trail():
    tracker=BallTracks()
    tracker.update(0,frame_with((30,50)),[item((30,50),99)])
    tracker.update(1,frame_with((35,50)),[item((35,50),99)])
    tracker.update(2,frame_with((40,50)),[])
    assert tracker.states[99]['status']=='lost'
    found=tracker.update(3,frame_with((45,50)),[item((45,50),99)])
    assert found[0]['track_id']==99 and found[0]['ball_id']==1
    assert tracker.states[99]['path'][-1][2] is True


def test_selected_id_cannot_force_wrong_appearance_match():
    tracker=BallTracks()
    tracker.update(0,frame_with((50,50)),[item((50,50),101)])
    blue=frame_with((150,50),(50,50))
    assert tracker.update(1,blue,[item((50,50),101)])==[]
    assert tracker.states[101]['status']=='lost'
    assert tracker.update(2,frame_with((50,50)),[item((50,50),101)])[0]['track_id']==101


def test_ambiguous_identical_crossing_is_lost_not_swapped():
    tracker=BallTracks()
    frame=frame_with((60,50));cv2.circle(frame,(80,50),8,(20,20,220),-1)
    tracker.update(0,frame,[item((60,50)),item((80,50))])
    # Equal motion costs and identical appearance give no reliable identity.
    found=tracker.update(1,frame_with((70,50)),[item((70,50))])
    assert found==[] and all(s['status']=='lost' for s in tracker.states.values())


def test_bounded_trails_and_retired_selected_id_not_recycled():
    tracker=BallTracks(max_points=3,max_age=2)
    for i in range(10):tracker.update(i,frame_with((30+2*i,50)),[item((30+2*i,50),42)])
    assert len(tracker.states[42]['path'])==3
    tracker.update(13,frame_with((50,50)),[])
    assert 42 not in tracker.states
    assert tracker.update(14,frame_with((50,50)),[item((50,50),42)])==[]


def test_position_recording_streams_mm_lost_and_session_reset(tmp_path):
    tracker=BallTracks();recorder=TrackRecorder(tmp_path)
    geometry=TableCoordinates([[0,0],[179,0],[179,99],[0,99]],TableDimensions(1790,990))
    tracker.update(0,frame_with((30,50)),[item((30,50),5)])
    recorder.record(tracker,0,0,30,geometry,'sample.mp4',TableDimensions().__dict__)
    tracker.update(1,frame_with((30,50)),[])
    recorder.record(tracker,1,1,30,geometry,'sample.mp4',TableDimensions().__dict__)
    path=recorder.path;recorder.close()
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]['type']=='session'
    assert rows[1]['tracks'][0]['center_mm']==[305.0,505.0]
    assert rows[2]['tracks'][0]['status']=='lost' and rows[2]['tracks'][0]['center_px'] is None
